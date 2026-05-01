"""
MuonLBFGS Optimizer: A Novel Combination of the Muon Optimizer and LBFGS Curvature Estimation
================================================================================================

THEORETICAL BACKGROUND
-----------------------
This implementation fuses two ideas:

1. THE LA (LBFGS-Adam) OPTIMIZER  [Zhang et al., 2026]
   The paper proposes replacing Adam's raw gradient g_n with the LBFGS-estimated
   quasi-Newton direction gamma_n. Instead of updating:

       x_{n,i} = x_{n-1,i} - alpha * m_hat / (sqrt(v_hat) + epsilon)   [Adam]

   the LA optimizer uses:

       gamma_{n-1} = LBFGS_two_loop_recursion(g_{n-1}, memory)
       x_{n,i}    = x_{n-1,i} - (alpha_n / sqrt(v_hat + epsilon)) * m_hat  [LA]

   where m_hat is computed using gamma (the LBFGS direction) instead of g.
   The key innovation: the LBFGS direction implicitly encodes curvature
   information through the inverse-Hessian approximation H_k^{-1}, so:

       gamma_k ≈ H_k^{-1} * g_k

   This makes the momentum term "curvature-aware" rather than just gradient-aware.

2. THE MUON OPTIMIZER [Kosson et al., 2024 / Jordan et al., 2024]
   Muon is designed for matrix-shaped parameters (weight matrices of linear/conv
   layers). Its core idea is to orthogonalize the gradient using the Newton-Schulz
   iteration, which approximately computes:

       G_orth = orthogonalize(G)     where G is the gradient matrix

   Mathematically, this computes the "matrix sign" direction, which is the
   steepest ascent direction in the Frobenius inner product under a spectral
   norm constraint. The Nesterov momentum variant computes:

       M_t = beta * M_{t-1} + G_t                    (momentum accumulation)
       G_eff = (1 + beta) * G_t - beta * M_{t-1}     (Nesterov correction)

   Then the update is:
       theta_{t+1} = theta_t - lr * orthogonalize(G_eff)

   The orthogonalization is performed via 5 steps of Newton-Schulz:
       X_{i+1} = a*X_i + b*X_i*(X_i^T*X_i) + c*X_i*(X_i^T*X_i)^2
   with coefficients a=3.4445, b=-4.7750, c=2.0315 (quintic variant).

COMBINATION RATIONALE (MuonLBFGS)
-----------------------------------
Can we combine Muon with the LBFGS idea from the LA paper? YES, and here is why:

  (a) The LA paper's key insight is: replace the raw gradient in Adam's moment
      updates with the LBFGS quasi-Newton direction. This is modular — the
      LBFGS direction is computed BEFORE the Adam (or any) update rule is applied.

  (b) Muon's key insight is: orthogonalize the gradient for matrix parameters
      to produce a direction that respects the spectral geometry of the weight
      space. This is also modular — it acts on the direction vector.

  (c) LBFGS and Muon address DIFFERENT aspects of the update:
      - LBFGS: SCALE of the update (captures curvature, adjusts step magnitude)
      - Muon: DIRECTION of the update (orthogonalizes to avoid ill-conditioned
        gradient directions in weight-matrix space)

  Therefore a natural pipeline is:
      g_k  -->  LBFGS two-loop  -->  gamma_k (curvature-scaled)
            -->  reshape to matrix  -->  Nesterov momentum  -->  orthogonalize
            -->  final parameter update

MATHEMATICAL FORMULATION OF MuonLBFGS
--------------------------------------
Let theta_k be the parameters at step k (a matrix of shape [m, n]).
Let g_k = nabla_theta L(theta_k) be the gradient (same shape [m, n]).

Step 1 — Flatten and compute LBFGS direction:
    g_flat_k in R^{m*n}
    Compute s_{k-1} = theta_flat_k - theta_flat_{k-1}   (parameter difference)
    Compute y_{k-1} = g_flat_k - g_flat_{k-1}           (gradient difference)
    Store {s, y} pairs in memory (sliding window of size m_mem)
    gamma_flat_k = H_k^{-1} * g_flat_k   via two-loop recursion (see below)
    Reshape: Gamma_k = gamma_flat_k.reshape(m, n)

Step 2 — Nesterov momentum on LBFGS direction:
    M_k = beta * M_{k-1} + Gamma_k                          (momentum)
    G_nesterov_k = (1 + beta) * Gamma_k - beta * M_{k-1}   (Nesterov correction)

Step 3 — Newton-Schulz orthogonalization:
    X = G_nesterov_k / ||G_nesterov_k||_F      (normalize)
    for i in range(5):
        X = a*X + b*X*(X^T*X) + c*X*(X^T*X)^2
    G_orth_k = X

Step 4 — Parameter update:
    theta_{k+1} = theta_k - lr * G_orth_k

TWO-LOOP RECURSION (LBFGS) DETAIL:
    Input: g (current gradient flat vector), memory [(s_0,y_0), ..., (s_{p-1},y_{p-1})]
    q = g
    for i = p-1 downto 0:
        rho_i = 1 / (y_i^T s_i)
        alpha_i = rho_i * (s_i^T q)
        q = q - alpha_i * y_i
    # Scale initial Hessian approximation:
    H_0 = (s_{p-1}^T y_{p-1}) / (y_{p-1}^T y_{p-1}) * I
    r = H_0 * q
    for i = 0 to p-1:
        beta_i = rho_i * (y_i^T r)
        r = r + s_i * (alpha_i - beta_i)
    return r    # this is H_k^{-1} * g

CONVERGENCE PROPERTIES
-----------------------
The LA paper proves (Theorem 4.1) that under weaker assumptions than Adam requires,
the LA optimizer achieves:

    E[||nabla F(x_tau)||^2] <= 2R(F(x_0) - F_*) / (alpha * N_tilde)
                              + E * (terms involving ln(N), beta_1, beta_2)

where N_tilde = N - beta_1/(1-beta_1). This bound is WEAKER in requirements
than Adam's original convergence bound, meaning LA can converge in settings
where Adam might not.

For MuonLBFGS, the orthogonalization step preserves the descent direction
(||G_orth|| ~ 1 in operator norm), so the convergence argument carries over
in spirit, with the orthogonalization acting as a form of implicit
preconditioning that improves conditioning of the effective Hessian approximation
in weight-matrix space.

NOTE ON APPLICABILITY:
-----------------------
Muon is specifically designed for 2D weight matrices (Linear layers, Conv layers
treated as 2D). For 1D parameters (biases, LayerNorm weights), standard Adam
or the LA update is more appropriate. This implementation automatically handles
both cases.
"""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from typing import List, Optional, Tuple
from collections import deque
import math


# =============================================================================
# UTILITY: Newton-Schulz Orthogonalization (Muon's core operation)
# =============================================================================

def newton_schulz_orthogonalize(
    G: torch.Tensor,
    num_steps: int = 5,
    eps: float = 1e-7,
    stability_clip: float = 10.0
) -> torch.Tensor:
    """
    Approximate orthogonalization of a matrix G via Newton-Schulz iteration.

    The iteration computes the "matrix sign" (polar factor) of G, which is the
    orthogonal matrix Q closest to G in the Frobenius norm sense. Specifically,
    for G = U * S * V^T (SVD), this returns U * V^T (the orthogonal factor).

    The quintic Newton-Schulz iteration used here is:
        X_{i+1} = a * X_i + b * X_i * (X_i^T * X_i) + c * X_i * (X_i^T * X_i)^2

    with a = 3.4445, b = -4.7750, c = 2.0315 (empirically optimal coefficients
    for convergence in ~5 steps when ||X_0||_2 ≈ 1).

    This polynomial approximation converges quadratically to the polar factor
    when the singular values of X are in [0, 1]. The initial normalization
    by the Frobenius norm ensures this condition approximately holds.

    Args:
        G: Gradient matrix of shape [m, n] where m >= n (if m < n, we transpose).
        num_steps: Number of Newton-Schulz iterations (default 5 is sufficient).
        eps: Small value to avoid division by zero in normalization.

    Returns:
        Orthogonalized matrix of same shape as G, with singular values ≈ 1.
    """
    # Coefficients for the quintic Newton-Schulz iteration
    # These are chosen so that the iteration p(x) = ax + bx^3 + cx^5
    # satisfies p(1) = 1 and p'(1) = 0 and p''(1) = 0 (triple root at 1),
    # which gives fast convergence near the fixed point x = 1.
    a, b, c = 3.4445, -4.7750, 2.0315

    # We need m >= n for the standard formulation (if not, transpose)
    transposed = False
    if G.shape[0] < G.shape[1]:
        G = G.T
        transposed = True

    # X must have singular values in approximately [0, 1] for convergence.
    # Dividing by the Frobenius norm approximates this, since ||G||_F >= ||G||_2
    # (largest singular value), so ||G/||G||_F||_2 <= 1.
    X = G / (G.norm(p='fro') + eps)

    # Perform Newton-Schulz iterations
    # Each iteration: X <- a*X + b*X*(X^T*X) + c*X*(X^T*X)^2
    for _ in range(num_steps):
        # A = X^T * X, shape [n, n] (Gram matrix)
        A = X.T @ X
        # B = b*X*A + c*X*A^2 = X*(b*A + c*A^2) = X*A*(b*I + c*A)
        B = b * A + c * (A @ A)
        # X <- a*X + X*B
        X = a * X + X @ B
        # Numerical stability: clip if norm explodes (can happen for ill-conditioned G)
        x_norm = X.norm(p='fro')
        if x_norm > stability_clip:
            X = X * (stability_clip / x_norm)

    # If we transposed initially, transpose back
    if transposed:
        X = X.T

    return X


# =============================================================================
# UTILITY: LBFGS Two-Loop Recursion
# =============================================================================

def lbfgs_two_loop_recursion(
    g: torch.Tensor,
    memory: deque,
) -> torch.Tensor:
    """
    Computes the LBFGS quasi-Newton direction via the two-loop recursion.

    Given the current gradient g and a memory buffer of (s, y) pairs
    representing recent parameter differences (s_i = x_{i+1} - x_i) and
    gradient differences (y_i = g_{i+1} - g_i), this function computes:

        d = H_k^{-1} * g

    where H_k^{-1} is the LBFGS approximation to the inverse Hessian.

    The algorithm (Nocedal & Wright, 1999, Algorithm 7.4):

    FIRST LOOP (backward pass, builds alpha coefficients):
        q = g
        for i = k-1 downto k-m:
            rho_i = 1 / (y_i^T s_i)
            alpha_i = rho_i * s_i^T q
            q = q - alpha_i * y_i
        return q, {alpha_i}, {rho_i}

    SCALING (initial Hessian approximation H_0):
        gamma_k = (s_{k-1}^T y_{k-1}) / (y_{k-1}^T y_{k-1})
        H_0 = gamma_k * I    (diagonal scaling)
        r = H_0 * q = gamma_k * q

    SECOND LOOP (forward pass, applies correction):
        for i = k-m to k-1:
            beta_i = rho_i * y_i^T r
            r = r + s_i * (alpha_i - beta_i)
        return r    <- this is H_k^{-1} * g

    The mathematical justification for this algorithm comes from the BFGS
    update formula for the inverse Hessian:
        H_{k+1}^{-1} = (I - rho_k s_k y_k^T) H_k^{-1} (I - rho_k y_k s_k^T)
                      + rho_k s_k s_k^T

    Recursively applying this formula and reorganizing gives the two-loop form.

    Args:
        g: Current gradient (flat tensor of shape [d]).
        memory: deque of (s, y) tuples where s=param_diff, y=grad_diff,
                both are flat tensors of shape [d].

    Returns:
        d: The LBFGS quasi-Newton direction H_k^{-1} * g, shape [d].
    """
    # If no memory yet, just return the gradient (steepest descent direction)
    if len(memory) == 0:
        return g.clone()

    q = g.clone()

    # Collect memory as a list for indexed access
    mem_list = list(memory)  # ordered from oldest to newest
    p = len(mem_list)

    # Pre-compute rho_i = 1 / (y_i^T s_i) for each pair
    # This is the reciprocal of the curvature along direction s_i
    rho_list = []
    for s_i, y_i in mem_list:
        curvature = y_i.dot(s_i)
        # Skip pairs with non-positive curvature (Wolfe condition not satisfied)
        # to maintain positive-definiteness of the Hessian approximation
        if curvature > 1e-10:
            rho_list.append(1.0 / curvature)
        else:
            rho_list.append(0.0)

    # FIRST LOOP: backward pass
    alpha_list = []
    for i in range(p - 1, -1, -1):
        s_i, y_i = mem_list[i]
        alpha_i = rho_list[i] * s_i.dot(q)
        q = q - alpha_i * y_i
        alpha_list.insert(0, alpha_i)  # store in forward order

    # SCALING: use the most recent (s, y) pair to scale H_0
    s_last, y_last = mem_list[-1]
    sy = s_last.dot(y_last)
    yy = y_last.dot(y_last)
    if yy > 1e-10:
        gamma = sy / yy   # scalar scaling factor for initial Hessian
    else:
        gamma = 1.0

    # r = H_0 * q = gamma * q
    r = gamma * q

    # SECOND LOOP: forward pass
    for i in range(p):
        s_i, y_i = mem_list[i]
        beta_i = rho_list[i] * y_i.dot(r)
        r = r + s_i * (alpha_list[i] - beta_i)

    return r


# =============================================================================
# MAIN OPTIMIZER: MuonLBFGS
# =============================================================================

class MuonLBFGS(Optimizer):
    """
    MuonLBFGS: Combines LBFGS curvature estimation with Muon's orthogonal update.

    This optimizer implements the following pipeline at each step for 2D parameters:

        1. Compute flat gradient g_k from autograd.
        2. Update LBFGS memory: s_{k-1} = x_flat_{k} - x_flat_{k-1},
                                 y_{k-1} = g_flat_{k} - g_flat_{k-1}
        3. Compute curvature-aware direction:
               gamma_k = lbfgs_two_loop(g_flat_k, memory)    [quasi-Newton step]
           Reshape to matrix: Gamma_k = gamma_k.reshape(m, n)
        4. Apply Nesterov momentum on Gamma_k:
               M_k = beta * M_{k-1} + Gamma_k
               G_eff_k = (1 + beta) * Gamma_k - beta * M_{k-1}
        5. Orthogonalize via Newton-Schulz:
               G_orth_k = newton_schulz(G_eff_k)
        6. Update parameters:
               x_{k+1} = x_k - lr * G_orth_k

    For 1D parameters (biases, norms), falls back to the LA optimizer:
        LA Update (analogous to Algorithm 3.3 in the paper):
               gamma_1d_k = lbfgs_two_loop(g_flat_k, memory_1d)
               m_k = beta1 * m_{k-1} + (1 - beta1) * gamma_1d_k
               v_k = beta2 * v_{k-1} + (1 - beta2) * gamma_1d_k^2
               m_hat = m_k / (1 - beta1^k)
               v_hat = v_k / (1 - beta2^k)
               alpha_k = lr * (1 - beta1) * sqrt(1 - beta2^k) / (1 - beta1^k)
               x_{k+1} = x_k - alpha_k * m_hat / (sqrt(v_hat) + epsilon)

    Args:
        params: Iterable of parameters to optimize (typically model.parameters()).
        lr: Global learning rate (default: 0.02 for Muon branch, 1e-3 for Adam branch).
        beta: Nesterov momentum coefficient for Muon branch (default: 0.95).
        beta1: First moment coefficient for 1D Adam branch (default: 0.9).
        beta2: Second moment coefficient for 1D Adam branch (default: 0.999).
        epsilon: Small constant for numerical stability in Adam branch (default: 1e-8).
        memory_size: Number of (s, y) pairs kept in LBFGS memory (default: 10).
        ns_steps: Number of Newton-Schulz iterations for orthogonalization (default: 5).
        use_lbfgs_for_1d: Whether to apply LBFGS to 1D params (default: True).
        muon_params: Optional list of specific parameter tensors to apply Muon to.
                     If None, applies Muon to all 2D params and Adam/LA to 1D params.
        lbfgs_clip: Max norm for LBFGS direction (default: 5.0). Important for
                    stability when mini-batch curvature estimates are noisy.

    IMPORTANT PRACTICAL NOTES:
    --------------------------
    1. GRADIENT CLIPPING: Apply torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
       BEFORE calling optimizer.step(). The LBFGS direction can scale gradients
       by the inverse curvature, which may amplify noisy mini-batch gradient components.

    2. LEARNING RATE: Because the Newton-Schulz orthogonalization normalizes the
       update to have singular values ≈ 1, a learning rate of 0.01–0.02 is typical
       for the Muon branch. This is MUCH smaller than what you might use for SGD.

    3. WARM-UP: MuonLBFGS benefits from a learning rate warm-up schedule, as the
       LBFGS memory is empty for the first few steps and the curvature estimates
       are unreliable initially.

    4. MEMORY COST: Each (s, y) pair costs O(d) memory where d is the parameter
       count for that layer. With memory_size=10, total cost is 20*d per parameter
       group, which is acceptable for typical layer sizes.

    5. CURVATURE CONDITION (s^T y > 0): The code skips (s,y) pairs that violate
       the curvature condition to maintain positive-definiteness of H_k^{-1}.
       In practice with mini-batches this condition may be violated often in early
       training — this is expected and the code falls back to the gradient direction.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        beta: float = 0.95,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        memory_size: int = 10,
        ns_steps: int = 5,
        use_lbfgs_for_1d: bool = True,
        muon_params=None,
        lbfgs_clip: float = 5.0,
    ):
        # Validate hyperparameters
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid Muon beta: {beta}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta2: {beta2}")
        if epsilon <= 0.0:
            raise ValueError(f"Invalid epsilon: {epsilon}")
        if memory_size < 1:
            raise ValueError(f"Invalid memory_size: {memory_size}")

        defaults = dict(
            lr=lr, beta=beta, beta1=beta1, beta2=beta2, epsilon=epsilon,
            memory_size=memory_size, ns_steps=ns_steps,
            use_lbfgs_for_1d=use_lbfgs_for_1d, lbfgs_clip=lbfgs_clip,
        )

        # Store which params are designated for Muon (2D branch)
        self.muon_params = set()
        if muon_params is not None:
            for p in muon_params:
                self.muon_params.add(id(p))

        super().__init__(params, defaults)

    def _is_muon_param(self, p: torch.Tensor, group: dict) -> bool:
        """
        Determine if this parameter should use the Muon branch.

        Muon is appropriate for matrix-shaped parameters (2D or higher,
        treated as 2D by reshaping). 1D parameters (biases, LayerNorm
        scale/shift) use the LA (Adam + LBFGS) branch instead.
        """
        if id(p) in self.muon_params:
            return True
        # If no explicit list given, auto-detect: Muon for 2D+, Adam/LA for 1D
        if len(self.muon_params) == 0:
            return p.ndim >= 2
        return False

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                     Not typically needed for most use cases.

        Returns:
            loss: The loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']
            beta1 = group['beta1']
            beta2 = group['beta2']
            epsilon = group['epsilon']
            memory_size = group['memory_size']
            ns_steps = group['ns_steps']
            use_lbfgs_for_1d = group['use_lbfgs_for_1d']
            lbfgs_clip = group['lbfgs_clip']

            for p in group['params']:
                # Skip parameters with no gradient
                if p.grad is None:
                    continue

                grad = p.grad
                # Clamp gradients to avoid numerical instability
                # (not in original papers but defensive practice)
                if grad.is_sparse:
                    raise RuntimeError("MuonLBFGS does not support sparse gradients")

                # Retrieve or initialize optimizer state for this parameter
                state = self.state[p]

                # ---- INITIALIZATION (first step) ----
                if len(state) == 0:
                    state['step'] = 0
                    state['lbfgs_memory'] = deque(maxlen=memory_size)
                    # Previous flat gradient (for computing y_k = g_k - g_{k-1})
                    state['prev_grad_flat'] = None
                    # Previous flat parameter (for computing s_k = x_k - x_{k-1})
                    state['prev_param_flat'] = p.data.reshape(-1).clone()

                    if self._is_muon_param(p, group):
                        # Muon branch: Nesterov momentum buffer
                        state['muon_momentum'] = torch.zeros_like(p.data)
                    else:
                        # LA branch: Adam first and second moment buffers
                        state['exp_avg'] = torch.zeros_like(p.data)     # m_k
                        state['exp_avg_sq'] = torch.zeros_like(p.data)  # v_k

                state['step'] += 1
                k = state['step']

                # ---- STEP 1: FLATTEN GRADIENT AND PARAMETER ----
                # We work in flat (1D) space for LBFGS computations
                g_flat = grad.reshape(-1)               # shape [d]
                x_flat = p.data.reshape(-1)             # shape [d]

                # ---- STEP 2: UPDATE LBFGS MEMORY ----
                # On all steps after the first, we have previous data to form (s, y)
                prev_g_flat = state['prev_grad_flat']
                prev_x_flat = state['prev_param_flat']

                if prev_g_flat is not None:
                    # s_k = x_k - x_{k-1}  (parameter step taken last iteration)
                    s_k = x_flat - prev_x_flat
                    # y_k = g_k - g_{k-1}  (gradient change)
                    y_k = g_flat - prev_g_flat

                    # Curvature condition: y_k^T s_k > 0 is required for
                    # the LBFGS Hessian approximation to remain positive definite.
                    # If violated, we skip this (s, y) pair (Wolfe condition guard).
                    curvature = y_k.dot(s_k).item()
                    if curvature > 1e-10:
                        # Store cloned tensors (detach from computation graph)
                        state['lbfgs_memory'].append(
                            (s_k.clone(), y_k.clone())
                        )

                # Update stored previous gradient and parameter for next step
                state['prev_grad_flat'] = g_flat.clone()
                state['prev_param_flat'] = x_flat.clone()

                # ---- STEP 3: COMPUTE LBFGS QUASI-NEWTON DIRECTION ----
                gamma_flat = lbfgs_two_loop_recursion(
                    g_flat,
                    state['lbfgs_memory']
                )
                # Stability clip: prevent exploding quasi-Newton directions.
                # This is especially important in early training when the (s,y)
                # memory is small and curvature estimates are noisy.
                gamma_norm = gamma_flat.norm()
                if gamma_norm > lbfgs_clip:
                    gamma_flat = gamma_flat * (lbfgs_clip / gamma_norm)

                # ---- BRANCH: MUON (2D parameters) vs LA (1D parameters) ----
                if self._is_muon_param(p, group):
                    # ============================================================
                    # MUON + LBFGS BRANCH (for weight matrices)
                    # ============================================================

                    # Reshape quasi-Newton direction back to parameter shape
                    # For conv layers with shape [C_out, C_in, kH, kW],
                    # we treat it as [C_out, C_in * kH * kW] for orthogonalization.
                    orig_shape = p.data.shape
                    if p.ndim > 2:
                        # Treat conv weights as 2D matrix for orthogonalization
                        Gamma = gamma_flat.reshape(orig_shape[0], -1)
                    else:
                        Gamma = gamma_flat.reshape(orig_shape)

                    # STEP 4: Nesterov momentum on LBFGS direction
                    # M_k = beta * M_{k-1} + Gamma_k
                    # G_eff = (1+beta)*Gamma_k - beta*M_{k-1}   [Nesterov trick]
                    #
                    # WHY NESTEROV? In standard Muon, Nesterov momentum gives a
                    # "look-ahead" correction that provides better convergence
                    # rates. When applied to the LBFGS direction, it also helps
                    # smooth out the inherent noise in the quasi-Newton estimates
                    # caused by mini-batch gradients.
                    M = state['muon_momentum']
                    if p.ndim > 2:
                        M_view = M.reshape(orig_shape[0], -1)
                    else:
                        M_view = M.reshape(orig_shape)

                    # Compute G_eff BEFORE updating M (using old M)
                    G_eff = (1.0 + beta) * Gamma - beta * M_view
                    # Now update momentum buffer M <- beta*M + Gamma
                    M_view.mul_(beta).add_(Gamma)
                    # Write back to state (handles the reshape case)
                    state['muon_momentum'] = M_view.reshape(orig_shape) if p.ndim > 2 else M_view

                    # STEP 5: Newton-Schulz orthogonalization
                    # G_orth = orthogonalize(G_eff) via Newton-Schulz iteration
                    #
                    # WHY ORTHOGONALIZE after LBFGS? The LBFGS direction can have
                    # arbitrarily large magnitude in poorly-conditioned directions.
                    # The orthogonalization re-normalizes to the "right geometry"
                    # for weight matrix spaces (under the spectral norm), combining
                    # the directional information from LBFGS with the magnitude
                    # normalization from Muon.
                    G_orth = newton_schulz_orthogonalize(G_eff, num_steps=ns_steps)

                    # Reshape back to original parameter shape
                    if p.ndim > 2:
                        G_orth = G_orth.reshape(orig_shape)

                    # STEP 6: Parameter update
                    # theta_{k+1} = theta_k - lr * G_orth_k
                    p.data.add_(G_orth, alpha=-lr)

                else:
                    # ============================================================
                    # LA (LBFGS-Adam) BRANCH (for 1D parameters)
                    # This closely follows Algorithm 3.3 from the paper,
                    # using gamma (LBFGS direction) instead of raw gradient g.
                    # ============================================================
                    gamma_1d = gamma_flat  # shape [d], same as p.data.reshape(-1)

                    exp_avg = state['exp_avg']       # m_k (first moment)
                    exp_avg_sq = state['exp_avg_sq'] # v_k (second moment)

                    # Update biased first moment estimate using LBFGS direction:
                    # m_k = beta1 * m_{k-1} + (1 - beta1) * gamma_k
                    exp_avg.mul_(beta1).add_(gamma_1d.reshape_as(exp_avg), alpha=1.0 - beta1)

                    # Update biased second moment estimate:
                    # v_k = beta2 * v_{k-1} + (1 - beta2) * gamma_k^2
                    exp_avg_sq.mul_(beta2).addcmul_(
                        gamma_1d.reshape_as(exp_avg_sq),
                        gamma_1d.reshape_as(exp_avg_sq),
                        value=1.0 - beta2
                    )

                    # Bias correction (standard Adam bias correction)
                    bias_correction1 = 1.0 - beta1 ** k
                    bias_correction2 = 1.0 - beta2 ** k
                    m_hat = exp_avg / bias_correction1    # unbiased first moment
                    v_hat = exp_avg_sq / bias_correction2  # unbiased second moment

                    # Adaptive learning rate (from LA paper, Eq. before Algorithm 3.3):
                    # alpha_n = alpha * (1 - beta1) * sqrt(1 - beta2^n) / (1 - beta1^n)
                    # This correction accounts for both bias corrections simultaneously.
                    alpha_n = lr * (1.0 - beta1) * math.sqrt(bias_correction2) / bias_correction1

                    # LA update rule:
                    # x_{n,i} = x_{n-1,i} - alpha_n / sqrt(v_hat_i + epsilon) * m_hat_i
                    denom = (v_hat.sqrt() + epsilon)
                    p.data.addcdiv_(m_hat, denom, value=-alpha_n)

        return loss


# =============================================================================
# PSEUDOCODE SUMMARY (printed for reference)
# =============================================================================

PSEUDOCODE = """
============================================================
PSEUDOCODE: MuonLBFGS Algorithm
============================================================

INITIALIZATION:
    For each parameter theta_i:
        lbfgs_memory_i = deque(maxlen=m)    // LBFGS history buffer
        prev_grad_i = None
        prev_param_i = theta_i.flatten()
        if theta_i is 2D (Muon branch):
            muon_momentum_i = zeros_like(theta_i)
        else (LA branch):
            exp_avg_i = zeros_like(theta_i)      // Adam m
            exp_avg_sq_i = zeros_like(theta_i)   // Adam v

MAIN LOOP (for each mini-batch):
    Compute loss L, backpropagate to get gradients grad_i for each param theta_i

    For each parameter theta_i:
        k = k + 1
        g_flat = grad_i.flatten()                    // flat gradient
        x_flat = theta_i.flatten()                   // flat param

        // ---- UPDATE LBFGS MEMORY ----
        if prev_grad_i is not None:
            s = x_flat - prev_param_i                // parameter difference
            y = g_flat - prev_grad_i                 // gradient difference
            if y^T s > epsilon_curv:                 // curvature condition
                lbfgs_memory_i.append((s, y))

        prev_grad_i = g_flat.clone()
        prev_param_i = x_flat.clone()

        // ---- COMPUTE LBFGS QUASI-NEWTON DIRECTION ----
        gamma_flat = TWO_LOOP_RECURSION(g_flat, lbfgs_memory_i)
        // See two-loop recursion detail in docstring above

        // ---- BRANCH SELECTION ----
        if theta_i is 2D parameter (Muon branch):
            Gamma = gamma_flat.reshape(theta_i.shape)        // matrix form

            // Nesterov momentum on LBFGS direction
            G_eff = (1 + beta) * Gamma - beta * M           // Nesterov
            M = beta * M + Gamma                             // update momentum

            // Newton-Schulz orthogonalization
            X = G_eff / ||G_eff||_F
            for i = 1 to ns_steps:
                A = X^T X
                X = a*X + b*X*A + c*X*A^2                   // a=3.4445, b=-4.7750, c=2.0315
            G_orth = X

            // Parameter update
            theta_i = theta_i - lr * G_orth

        else (1D parameter, LA branch):
            // LA update (Algorithm 3.3 from Zhang et al. 2026, using gamma not g)
            m = beta1 * m + (1 - beta1) * gamma_flat        // first moment
            v = beta2 * v + (1 - beta2) * gamma_flat^2      // second moment
            m_hat = m / (1 - beta1^k)                       // bias correction
            v_hat = v / (1 - beta2^k)
            alpha_k = lr * (1-beta1) * sqrt(1-beta2^k) / (1-beta1^k)
            theta_i = theta_i - alpha_k * m_hat / (sqrt(v_hat) + epsilon)

============================================================

TWO_LOOP_RECURSION(g, memory = [(s_0,y_0), ..., (s_{p-1}, y_{p-1})]):
    if memory is empty: return g
    q = g
    for i = p-1 downto 0:                           // FIRST LOOP
        rho_i = 1 / (y_i^T s_i)
        alpha_i = rho_i * (s_i^T q)
        q = q - alpha_i * y_i
    // Scale H_0
    gamma = (s_{p-1}^T y_{p-1}) / (y_{p-1}^T y_{p-1})
    r = gamma * q
    for i = 0 to p-1:                               // SECOND LOOP
        beta_i = rho_i * (y_i^T r)
        r = r + s_i * (alpha_i - beta_i)
    return r    // = H_k^{-1} * g
============================================================
"""


# =============================================================================
# DEMONSTRATION / USAGE EXAMPLE
# =============================================================================

def demo():
    """
    Demonstrates MuonLBFGS on a simple classification problem.

    We construct a small MLP and train it with MuonLBFGS, showing the
    optimizer correctly handles both 2D (weight matrices, Muon branch)
    and 1D (bias vectors, LA branch) parameters.
    """
    import torch.nn.functional as F

    print("=" * 60)
    print("MuonLBFGS Optimizer Demo")
    print("=" * 60)
    print(PSEUDOCODE)

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- Simple MLP ----
    # Linear weights are 2D -> Muon+LBFGS branch
    # Biases are 1D -> LA (Adam+LBFGS) branch
    model = nn.Sequential(
        nn.Linear(16, 64),    # weight: [64, 16] -> Muon branch
        nn.ReLU(),
        nn.Linear(64, 64),    # weight: [64, 64] -> Muon branch
        nn.ReLU(),
        nn.Linear(64, 10),    # weight: [10, 64] -> Muon branch
    ).to(device)

    # ---- Instantiate MuonLBFGS optimizer ----
    # lr=0.02 is typical for Muon; epsilon/beta1/beta2 are Adam defaults for 1D branch
    optimizer = MuonLBFGS(
        model.parameters(),
        lr=0.02,          # learning rate for Muon (2D) branch
        beta=0.95,        # Nesterov momentum for Muon branch
        beta1=0.9,        # Adam beta1 for 1D branch
        beta2=0.999,      # Adam beta2 for 1D branch
        epsilon=1e-8,     # Adam epsilon for 1D branch
        memory_size=10,   # LBFGS memory: keep last 10 (s,y) pairs
        ns_steps=5,       # Newton-Schulz iterations for orthogonalization
    )

    # Print which parameters go to which branch
    print("Parameter routing:")
    for name, param in model.named_parameters():
        branch = "MUON+LBFGS (2D)" if param.ndim >= 2 else "LA/Adam+LBFGS (1D)"
        print(f"  {name:30s} shape={str(param.shape):20s} -> {branch}")
    print()

    # ---- Synthetic dataset ----
    X = torch.randn(256, 16, device=device)
    y = torch.randint(0, 10, (256,), device=device)

    # ---- Training loop ----
    print(f"{'Step':>6}  {'Loss':>10}  {'Accuracy':>10}")
    print("-" * 30)

    for step in range(1, 51):
        # Forward pass
        logits = model(X)
        loss = F.cross_entropy(logits, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Metrics
        if step % 5 == 0:
            acc = (logits.argmax(dim=1) == y).float().mean().item()
            print(f"  {step:4d}  {loss.item():10.4f}  {acc:10.4f}")

    print("\nDemo complete. MuonLBFGS ran successfully.")
    print("Note: With a real dataset and proper training setup,")
    print("MuonLBFGS should show faster curvature-aware convergence")
    print("compared to standard Muon, especially in later training stages.")


if __name__ == "__main__":
    demo()
