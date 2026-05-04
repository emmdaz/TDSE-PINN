class PINN:
    def __init__(self) -> None:
        self.net = DNN(dim_in = 2, dim_out = 2, n_layer = 10, n_node = 256, ub = ub, lb = lb).to(
            device
        )
        # Optimizer LBFGS for precision
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr = 1.0,
            max_iter = 50000, # Maximum iterations
            max_eval = 50000,
            tolerance_grad = 1e-5, # Minimum value for the gradient
            tolerance_change = 1.0 * np.finfo(float).eps,
            history_size = 50,
            line_search_fn = "strong_wolfe",
        )
        
        # Optimizers configuration
        """
        We have to consider two optimization processes since Muon is strictly designed 
        to work with 2D matrixes and the biases are 1D. 
        So in this case we'll have to optimize the weight matrixes (which indeed are 2D)
        using Muon; and in the other hand we'll have to use Adam or other optimizer in 
        order to deal with the biases."
        """
        # Separate the 2D and 1D params
        self.matrix_params = []
        self.vector_params = []
        
        for p in self.net.parameters():
            if p.ndim == 2:
                self.matrix_params.append(p)
            else:
                self.vector_params.append(p)
                
        # Apply the correspondent optimizer        
        self.muon = torch.optim.Muon(self.matrix_params, lr = 1e-3)
        self.adam = torch.optim.Adam(self.vector_params, lr = 1e-3)
        
        # Define the loss function (Since it is a PINN we'll use MSE)
        self.loss_fn = torch.nn.MSELoss()
        # An error for each term of the PINN loss
        self.losses = {"ic": [], "bc": [], "pde": []}
        # For iterations
        self.iter = 0
        
    # To split the output. We set the entire TDSE to be uv (Phi(x,t) = u(x,t) + i*v(x,t)) and then
    # it has to be separed into the real and imaginary part.
    def net_uv(self, xt):
        uv = self.net(xt)
        u = uv[:, 0:1] # u(x,t) 
        v = uv[:, 1:2] # v(x,t)
        return u, v

    def ic_loss(self, xt): # Initial condition loss
        uv_ic_pred = self.net(xt)
        mse_ic = self.loss_fn(uv_ic_pred, uv_ic)
        return mse_ic

    def bc_loss(self, xt_lb, xt_ub): # Boundary condition loss
        xt_lb = xt_lb.clone()
        xt_ub = xt_ub.clone()
        xt_lb.requires_grad = True
        xt_ub.requires_grad = True

        u_lb, v_lb = self.net_uv(xt_lb) # Evaluate the net in the lower boundary condition 
        u_ub, v_ub = self.net_uv(xt_ub) # Evaluate the net in the upper boundary ondition

        mse_bc1_u = self.loss_fn(u_lb, u_ub) # Boundary condition loss for the real part
        mse_bc1_v = self.loss_fn(v_lb, v_ub) # Boundary condition loss for the imaginary part

        u_x_lb = grad(u_lb.sum(), xt_lb, create_graph = True)[0][:, 0:1]
        u_x_ub = grad(u_ub.sum(), xt_ub, create_graph = True)[0][:, 0:1]

        v_x_lb = grad(v_lb.sum(), xt_lb, create_graph = True)[0][:, 0:1]
        v_x_ub = grad(v_ub.sum(), xt_ub, create_graph = True)[0][:, 0:1]
        mse_bc2_u = self.loss_fn(u_x_lb, u_x_ub)
        mse_bc2_v = self.loss_fn(v_x_lb, v_x_ub)

        mse_bc = mse_bc1_u + mse_bc1_v + mse_bc2_u + mse_bc2_v
        return mse_bc

    def pde_loss(self, xt): # PDE loss 
        xt = xt.clone()
        xt.requires_grad = True
        u, v = self.net_uv(xt)

        u_xt = grad(u.sum(), xt, create_graph=True)[0]
        u_x = u_xt[:, 0:1]
        u_xx = grad(u_x.sum(), xt, create_graph=True)[0][:, 0:1]
        u_t = u_xt[:, 1:2]

        v_xt = grad(v.sum(), xt, create_graph=True)[0]
        v_x = v_xt[:, 0:1]
        v_xx = grad(v_x.sum(), xt, create_graph=True)[0][:, 0:1]
        v_t = v_xt[:, 1:2]

        f_u = v_t - 0.5 * u_xx - (u ** 2 + v ** 2) * u  # u
        f_v = u_t + 0.5 * v_xx + (u ** 2 + v ** 2) * v  # v
        f_target = torch.zeros_like(f_u)

        mse_pde = self.loss_fn(f_u, f_target) + self.loss_fn(f_v, f_target)
        return mse_pde

    def closure(self):
        self.lbfgs.zero_grad()
        
        self.adam.zero_grad()
        self.muon.zero_grad()
        
        mse_ic = self.ic_loss(xt_ic)
        mse_bc = self.bc_loss(xt_lb, xt_ub)
        mse_pde = self.pde_loss(xt_f)
        loss = mse_ic + mse_bc + mse_pde
        loss.backward()

        self.losses["ic"].append(mse_ic.detach().cpu().item())
        self.losses["bc"].append(mse_bc.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        self.iter += 1
        print(
            f"\r{self.iter} Loss: {loss.item():.5e} IC: {mse_ic.item():.3e} BC: {mse_bc.item():.3e} pde: {mse_pde.item():.3e}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")
        return loss