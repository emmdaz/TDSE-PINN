"""
Multi-scale Momentum Muon (M3) Optimizer — PyTorch implementation
===================================================================

Faithful implementation of Algorithm 1, Section 7.2 of:
  "Nested Learning" (Behrouz et al., NeurIPS 2025)

Algorithm 1 — exact transcription from the paper
--------------------------------------------------
Inputs : Theta_0, L(.), eta, T (NS steps),
         beta1, beta2, beta3, alpha >= 0, eps, frequency f

1.  Init: M1_0 = M2_0 = V_0 = 0

2.  for k = 0, 1, 2, ...             [outer / low-frequency loop]

3.      M2_t = M2_{t-1} + beta3 * SUM_{i=(k-1)f}^{kf}  g_i
            (slow memory: accumulate the SUM of all f gradients
             in the just-completed chunk into running M2)

4.      O2_t = NewtonSchulz_T( M2_t )

5.      for t = k*f+1, ..., (k+1)*f  [inner / high-frequency loop]

6.          g_t = nabla_Theta L(Theta_t)

7.          M1_t = M1_{t-1} + beta1 * g_t      (cumulative sum)

8.          V_t  = V_{t-1}  + beta2 * g_t^2    (cumulative sum)

9.          O1_t = NewtonSchulz_T( M1_t )

10.         Theta_t = Theta_{t-1}
                    - eta * (O1_t + alpha * O2_t) / sqrt(V_t + eps)

11.     end for

12. end for

Critical design points
-----------------------
1. M1, M2, V are CUMULATIVE SUMS, NOT exponential moving averages.
   The paper writes  M <- M + beta*g,  not  M <- beta*M + g.
   This is a fundamental difference from Adam. As steps accumulate,
   the denominator sqrt(V) grows, providing implicit learning-rate
   decay. This means a larger initial lr (~0.01–0.1) is appropriate,
   unlike Adam where lr~1e-3 is standard.

2. M2 and O2 are computed ONCE per chunk of f steps and held constant
   for the entire inner loop. A gradient accumulator g_accum collects
   the sum of f gradients per chunk and is reset each outer iteration.

3. The global step counter increments ONCE per optimizer.step() call,
   not once per parameter. This is the most critical implementation
   detail — the TensorFlow Keras version incremented it P times per
   step (once per trainable variable), destroying the two-timescale
   structure.

4. Newton-Schulz at T=5 gives an approximate orthogonal direction
   (cosine similarity > 0.8 with a fully orthogonalised result). Full
   machine-precision convergence requires T≈20. For optimiser use T=5
   is a practical trade-off, exactly as the paper specifies.

5. Newton-Schulz is applied only to 2-D+ parameters (weight matrices,
   conv kernels). Biases and LayerNorm parameters are updated with raw
   momentum — orthogonalisation is not meaningful for 1-D tensors.
"""

import math
import functools
try:
    import torch
    from torch.optim import Optimizer
    _TORCH_AVAILABLE = True
    _no_grad = torch.no_grad
except ImportError:
    _TORCH_AVAILABLE = False

    def _no_grad():
        """No-op decorator factory — mirrors torch.no_grad() interface."""
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper
        return decorator

    class Optimizer:
        """Stub so M3Optimizer(Optimizer) parses without torch installed."""
        def __init__(self, params, defaults):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Newton-Schulz orthogonalisation
# ─────────────────────────────────────────────────────────────────────────────

def newton_schulz(M, steps: int = 5) :
    """
    Approximately orthogonalise a 2-D matrix M via the cubic iteration:

        X_{i+1} = 0.5 * X_i * (3*I - X_i^T * X_i)

    Derived from one gradient-descent step on ||O^T O - I||_F^2
    (Equations 43-44 of the NL paper). Converges to the orthogonal
    polar factor of M.

    Convergence notes:
    - After Frobenius normalisation, singular values of X are in ~[0, 0.6].
    - The cubic iteration slowly lifts them toward 1.
    - At T=5:  ||O^T O - I||_F ≈ 0.1–1.0  (approximate, good for optimiser)
    - At T=20: ||O^T O - I||_F ≈ 1e-15    (machine precision)
    - T=5 is a deliberate speed/accuracy trade-off per the paper.

    Args:
        M:     2-D float tensor, shape (m, n) with m >= n.
        steps: Cubic Newton-Schulz iterations. Paper uses T=5.

    Returns:
        Tensor O of same shape, with O^T O ≈ I_n.
    """
    assert M.ndim == 2, \
        f"newton_schulz requires 2-D input, got {M.ndim}-D"
    assert M.shape[0] >= M.shape[1], \
        f"newton_schulz requires rows >= cols, got shape {M.shape}. " \
        "Call orthogonalize() which handles the transpose automatically."

    # Normalise so singular values start near 0.5 (convergence basin).
    norm = M.norm() + 1e-8
    X    = M / norm

    n = X.shape[1]
    I = torch.eye(n, dtype=X.dtype, device=X.device)

    for _ in range(steps):
        A = X.t() @ X          # n × n
        X = 0.5 * X @ (3.0 * I - A)

    return X


def orthogonalize(T, steps: int = 5) :
    """
    Apply Newton-Schulz to a tensor of any shape.

    Handles:
      - ndim < 2  (scalars, bias vectors): returned unchanged.
      - ndim == 2 (weight matrices): orthogonalised directly.
      - ndim >= 3 (conv kernels):    all-but-last dims flattened to rows.
      - wide matrices (rows < cols): transposed before NS, transposed back.

    Args:
        T:     Input tensor of any shape.
        steps: Newton-Schulz iterations (T=5 per paper).

    Returns:
        Tensor of same shape with orthogonalised update direction,
        or T unchanged for ndim < 2.
    """
    if T.ndim < 2:
        return T

    original_shape = T.shape
    rows = math.prod(original_shape[:-1])
    cols = original_shape[-1]
    M    = T.reshape(rows, cols)

    # Transpose trick: NS requires rows >= cols.
    transposed = (rows < cols)
    if transposed:
        M = M.t()          # now (cols, rows)

    O = newton_schulz(M, steps=steps)

    if transposed:
        O = O.t()          # back to (rows, cols)

    return O.reshape(original_shape)


# ─────────────────────────────────────────────────────────────────────────────
# M3 Optimizer
# ─────────────────────────────────────────────────────────────────────────────

class M3Optimizer(Optimizer):
    """
    Multi-scale Momentum Muon (M3) Optimizer.

    Algorithm 1, Section 7.2 of "Nested Learning" (Behrouz et al., NeurIPS 2025).
    Conceptually: M3 = Adam + Muon + Continuum Memory System (CMS).

    Per-parameter optimizer state
    --------------------------------
    m1      : fast momentum   — cumulative sum M1_{t-1} + beta1 * g_t
    m2      : slow momentum   — cumulative sum, updated once per f steps
    v       : second moment   — cumulative sum V_{t-1}  + beta2 * g_t^2
    g_accum : gradient bucket — collects gradients within current chunk;
                                flushed into m2 at the next chunk boundary
    o2      : cached NewtonSchulz(m2); constant within a chunk

    Shared optimizer state
    -----------------------
    _global_step : int, incremented ONCE per optimizer.step() call.
                   Chunk boundaries occur when _global_step % frequency == 0.

    Args
    ----
    params    : Model parameters or param groups.
    lr        : Learning rate eta. Suggested range: 0.01–0.1.
                (Larger than Adam because cumulative-sum momentum has
                 implicit decay that requires a larger initial step.)
    beta1     : Fast momentum factor (default 0.9).
    beta2     : Second moment factor (default 0.999).
    beta3     : Slow momentum factor (default 0.9).
    alpha     : Weight of slow memory O2 in the update (default 0.1).
    eps       : Numerical stability constant (default 1e-8).
    frequency : Chunk size f — steps between M2 updates (default 10).
    ns_steps  : Newton-Schulz iterations T (default 5).
    """

    def __init__(
        self,
        params,
        lr: float      = 0.01,
        beta1: float   = 0.9,
        beta2: float   = 0.999,
        beta3: float   = 0.9,
        alpha: float   = 0.1,
        eps: float     = 1e-8,
        frequency: int = 10,
        ns_steps: int  = 5,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not (0.0 <= beta1 < 1.0):
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if not (0.0 <= beta3 < 1.0):
            raise ValueError(f"beta3 must be in [0, 1), got {beta3}")
        if alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        if frequency < 1:
            raise ValueError(f"frequency must be >= 1, got {frequency}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")

        defaults = dict(
            lr=lr, beta1=beta1, beta2=beta2, beta3=beta3,
            alpha=alpha, eps=eps, frequency=frequency, ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

        # THE critical implementation detail: a single integer shared across
        # all parameters, incremented exactly once per optimizer.step() call.
        # If this lived inside update logic (once per param), chunk detection
        # would fire P times per step instead of once — destroying the
        # two-timescale structure.
        self._global_step: int = 0

    @_no_grad()
    def step(self, closure=None):
        """
        Perform one M3 optimizer step.

        Standard usage:
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

        With closure:
            optimizer.step(closure)

        Returns:
            Scalar loss if closure is provided, else None.

        Why @torch.no_grad():
            Without this decorator PyTorch's autograd engine is active
            during the update, and any in-place write to a leaf parameter
            that requires_grad=True raises a RuntimeError. The decorator
            suspends gradient tracking for the entire method body, which
            is the correct behaviour for all optimiser step functions.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():   # re-enable inside closure only
                loss = closure()

        # ── Outer-loop: chunk-boundary detection ──────────────────────────
        # At the start of each new chunk (every f steps), we:
        #   (a) Flush g_accum into M2  (line 3 of Algorithm 1).
        #   (b) Recompute O2 = NS(M2) (line 4).
        #   (c) Zero g_accum for the new chunk.
        # At step 0 (first ever call), g_accum is zero so M2 stays at zero;
        # the first meaningful slow update fires at step f.
        is_slow_step = (self._global_step % self.defaults["frequency"] == 0)

        for group in self.param_groups:
            lr       = group["lr"]
            beta1    = group["beta1"]
            beta2    = group["beta2"]
            beta3    = group["beta3"]
            alpha    = group["alpha"]
            eps      = group["eps"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad  # gradient tensor (already computed by .backward())

                # ── Lazy state initialisation (first call only) ────────────
                state = self.state[p]
                if len(state) == 0:
                    state["m1"]      = torch.zeros_like(p)  # fast momentum
                    state["m2"]      = torch.zeros_like(p)  # slow momentum
                    state["v"]       = torch.zeros_like(p)  # second moment
                    state["g_accum"] = torch.zeros_like(p)  # chunk accumulator
                    state["o2"]      = torch.zeros_like(p)  # cached slow output

                m1      = state["m1"]
                m2      = state["m2"]
                v       = state["v"]
                g_accum = state["g_accum"]

                # ── Lines 3-4: slow memory (outer loop, once per chunk) ────
                if is_slow_step:
                    # Line 3: M2_t = M2_{t-1} + beta3 * SUM_{chunk} g_i
                    # g_accum already holds the sum of all f gradients from
                    # the just-completed chunk (filled below in the inner loop).
                    m2.add_(beta3 * g_accum)

                    # Line 4: O2_t = NewtonSchulz_T(M2_t)
                    if p.ndim >= 2:
                        state["o2"] = orthogonalize(m2, steps=ns_steps)
                    else:
                        # Biases / LayerNorm params: no orthogonalisation.
                        state["o2"] = m2.clone()

                    # Reset accumulator for the incoming chunk.
                    g_accum.zero_()

                # ── Lines 7-10: inner loop (every step) ───────────────────

                # Line 7: M1_t = M1_{t-1} + beta1 * g_t
                m1.add_(beta1 * g)

                # Line 8: V_t = V_{t-1} + beta2 * g_t^2
                v.add_(beta2 * g.square())

                # Gradient accumulation for the NEXT chunk's slow update.
                # This builds  SUM_{i=(k-1)f}^{kf} g_i  needed by line 3.
                g_accum.add_(g)

                # Line 9: O1_t = NewtonSchulz_T(M1_t)
                if p.ndim >= 2:
                    o1 = orthogonalize(m1, steps=ns_steps)
                else:
                    o1 = m1     # 1-D params: use raw momentum

                # Line 10: Theta_t = Theta_{t-1} - eta * (O1 + alpha*O2)
                #                                         / sqrt(V + eps)
                # Use p.data to write directly to the underlying storage,
                # bypassing autograd even if @torch.no_grad() is somehow
                # not active (e.g. inside a custom training loop that
                # wraps step() in torch.enable_grad()). This is the same
                # pattern used by torch.optim.Adam, SGD, AdamW, etc.
                o2 = state["o2"]
                update = lr * (o1 + alpha * o2) / (v + eps).sqrt()
                p.data.add_(-update)

        # ── Advance counter exactly once per optimizer.step() call ────────
        self._global_step += 1

        return loss

    # ── Serialisation: preserve chunk-boundary state across checkpoints ───

    def state_dict(self) -> dict:
        """
        Includes _global_step so chunk-boundary logic survives
        save/load cycles. Call torch.save(optimizer.state_dict(), path)
        as usual.
        """
        sd = super().state_dict()
        sd["_global_step"] = self._global_step
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        """Restores _global_step alongside all per-parameter states."""
        self._global_step = state_dict.pop("_global_step", 0)
        super().load_state_dict(state_dict)

    def __repr__(self) -> str:
        d = self.defaults
        return (
            f"M3Optimizer("
            f"lr={d['lr']}, beta1={d['beta1']}, beta2={d['beta2']}, "
            f"beta3={d['beta3']}, alpha={d['alpha']}, eps={d['eps']}, "
            f"frequency={d['frequency']}, ns_steps={d['ns_steps']})"
        )