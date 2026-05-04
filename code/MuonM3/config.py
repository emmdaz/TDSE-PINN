import torch
import torch.nn as nn # Layers and all neural network stuff
import torch.nn.functional as F # Related to the activation functions
import torch.optim as optim # These are the optimizers implemented in torch
from torch.autograd import grad # To compute derivatives via automatic differentiation

from pydoe import lhs # Latin Hypercube Sampling
import numpy as np

import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_float32_matmul_precision('highest')

# Setting a seed for replicate data
torch.manual_seed(1234)
np.random.seed(1234)

# To create layers in PyTorch
class layer(nn.Module):
    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x

# To create a dense neural network
# Recall that the activation function has to be Tanh since it is differentiable in all points
class DNN(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.ModuleList() # To being able to set a variable number of layers (since we'll use a loop)
        self.net.append(layer(dim_in, n_node, activation)) # Dim corresponding to the input size and n_node
                                                           # corresponds to the units for the first layer
        for _ in range(n_layer):
            self.net.append(layer(n_node, n_node, activation)) # All hidden layers have the same quantity of units
        self.net.append(layer(n_node, dim_out, activation = None)) # Last linear layer, we set the output dim
        self.ub = torch.tensor(ub, dtype = torch.float).to(device) 
        self.lb = torch.tensor(lb, dtype = torch.float).to(device)
        self.net.apply(weights_init)  

    def forward(self, x):
        x = (x - self.lb) / (self.ub - self.lb)  # Min-max scaling
        out = x
        for layer in self.net:
            out = layer(out)
        return out

def weights_init(m): # xavier initialization
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

def plotLoss(losses_dict, path, info=["IC", "BC", "PDE"]):
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(10, 6))
    axes[0].set_yscale("log")
    for i, j in zip(range(3), info):
        axes[i].plot(losses_dict[j.lower()])
        axes[i].set_title(j)
    plt.show()
    fig.savefig(path)

# Intervals for x and t
x_min = -5.0
x_max = 5.0
t_min = 0
t_max = np.pi / 2

# Then we set the boundary points
ub = np.array([x_max, t_max]) # Upper boundary point; x in (-5.0, 5.0)
lb = np.array([x_min, t_min]) # Lower boundary point; t in (0, pi/2)

# Number of points for the initial condition
N_ic = 50
# Number of points for the boundary condition
N_bc = 50
# Number of points for the collocation points (points sampled randonmly used to conserve the physics from the equation)
N_f = 20000

def trainingData():
    # Initial Conditions
    x_ic = np.random.uniform(x_min, x_max, (N_ic, 1)) # Create a set of random x used to compute the initial condition
    t_ic = np.zeros((N_ic, 1)) # Since all the points of x are evaluated in t = 0 we create a set full of zeros 
    xt_ic = np.hstack([x_ic, t_ic]) # All the (x,0) points for the initial condition
    
    # Here we separate the real part from the imaginary part of the Schrödinger equation
    # Let the real part be u(x,t) and the imaginary part v(x,t). We call phi(x,t) the Schrödinger time dependent 
    # equation being phi(x,t) = u(x,t) + i*v(x,t)
    
    # Then we set the initial conditions for each part
    u_ic = 2 * 2 / (np.exp(x_ic) + np.exp(-x_ic))  # 2 * sech(x), so the initial contition is u(x,0) = 2sech(x)
    v_ic = np.zeros((N_ic, 1)) # This way v(x,0) = 0 
    uv_ic = np.hstack([u_ic, v_ic]) # Then we put all togeteher to get a set with the values of the initial conditions for the 
    # TDSE complete (with the real and imaginary part)

    # Boundary conditions
    t_b = np.random.uniform(t_min, t_max, (N_bc, 1)) # Interval of t points
    x_lb = np.ones((N_bc, 1)) * x_min # Points for x = -0.5
    xt_lb = np.hstack([x_lb, t_b]) # Set of points to be used to evaluate the lower boundary points (x = -0.5, t)

    x_ub = np.ones((N_bc, 1)) * x_max # Points for x = 0.5
    xt_ub = np.hstack([x_ub, t_b]) # Set of points to be used to evaluate the upper boundary points (x = 0.5, t)

    # Parcial Differential Equation
    
    xt_f = lb + (ub - lb) * lhs(2, N_f) # lhs is a random points sample
    # By computing the previous line we get the collocations points. 
    xt_f = np.vstack([xt_ic, xt_lb, xt_ub, xt_f])
    
    # We get all points together a set which includes:
    # xt_ic - All points for the initial conditions
    # xt_lb - All points for the lower boundary points
    # xt_ub - All points for the upper boundary points
    # xt_f - All points for the collocations points
    # This set is created in order to get a set of collocation points used to preserve the physics and make a PINN

    # Tensor convertion
    xt_ic = torch.tensor(xt_ic, dtype=torch.float).to(device)
    uv_ic = torch.tensor(uv_ic, dtype=torch.float).to(device)
    xt_lb = torch.tensor(xt_lb, dtype=torch.float).to(device)
    xt_ub = torch.tensor(xt_ub, dtype=torch.float).to(device)
    xt_f = torch.tensor(xt_f, dtype=torch.float).to(device)

    return xt_ic, uv_ic, xt_lb, xt_ub, xt_f

xt_ic, uv_ic, xt_lb, xt_ub, xt_f = trainingData()