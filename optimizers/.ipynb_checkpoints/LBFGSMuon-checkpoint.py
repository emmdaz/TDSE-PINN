import torch
from torch.optim import Optimizer

class MuonLBFGS(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, history_size=10, eps=1e-8):
        defaults = dict(lr=lr, beta=beta, history_size=history_size, eps=eps)
        super().__init__(params, defaults)

        self.state['s_history'] = []
        self.state['y_history'] = []

    def two_loop_recursion(self, grad):
        """Compute L-BFGS direction"""
        s_hist = self.state['s_history']
        y_hist = self.state['y_history']

        q = grad.clone()
        alpha = []

        # Backward loop
        for s, y in reversed(list(zip(s_hist, y_hist))):
            rho = 1.0 / (y.dot(s) + 1e-10)
            a = rho * s.dot(q)
            alpha.append(a)
            q -= a * y

        # Initial Hessian scaling
        if len(s_hist) > 0:
            s, y = s_hist[-1], y_hist[-1]
            gamma = (s.dot(y) / (y.dot(y) + 1e-10))
            q *= gamma

        # Forward loop
        for (s, y), a in zip(zip(s_hist, y_hist), reversed(alpha)):
            rho = 1.0 / (y.dot(s) + 1e-10)
            b = rho * y.dot(q)
            q += s * (a - b)

        return q

    def step(self, closure):
        loss = closure()

        for group in self.param_groups:
            beta = group['beta']
            lr = group['lr']
            eps = group['eps']
            m_hist = group['history_size']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data.view(-1)

                state = self.state.setdefault(p, {})
                
                # Initialize momentum
                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(grad)
                    state['prev_param'] = p.data.view(-1).clone()
                    state['prev_grad'] = grad.clone()

                # === L-BFGS direction ===
                direction = self.two_loop_recursion(grad)

                # === Muon-style momentum update ===
                momentum = state['momentum']
                momentum.mul_(beta).add_(direction, alpha=(1 - beta))

                # === Update parameters ===
                update = lr * momentum
                p.data.add_(-update.view_as(p.data))

                # === Update L-BFGS memory ===
                s = p.data.view(-1) - state['prev_param']
                y = grad - state['prev_grad']

                if s.dot(y) > 1e-10:  # curvature condition
                    self.state['s_history'].append(s)
                    self.state['y_history'].append(y)

                    if len(self.state['s_history']) > m_hist:
                        self.state['s_history'].pop(0)
                        self.state['y_history'].pop(0)

                # Save previous
                state['prev_param'] = p.data.view(-1).clone()
                state['prev_grad'] = grad.clone()

        return loss