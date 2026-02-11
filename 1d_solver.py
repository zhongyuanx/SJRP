import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from networks import Vnet, Znet

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32

#Hyperparameters
num_sample = 1250 # number of sample paths to generate
T = 0.8 # total time horizon
N = 200 # number of time intervals to partition the time horizon
lam = 0.4 # rate of ordering events
num_iter = 15000 # number of iterations for the solver
learning_rate = 1e-3 # learning rate for the optimizer
beta = np.array([1e2, 1e4, 2*1e5]) #penalty parameter 
ln_mean = 2.0 # mean of the lognormal distribution for the reference process
ln_var = 1.0 # variance of the lognormal distribution for the reference process
ln_mu = np.log(ln_mean**2 / np.sqrt(ln_var + ln_mean**2)) # mu parameter of the lognormal
ln_sigma = np.sqrt(np.log(ln_var / ln_mean**2 + 1)) # sigma parameter of the lognormal

# Problem parameters
r = 0.05 # interest rate
p = 2.0 # penalty cost for backlogging
h = 0.5 # holding cost per unit per time step
c = 1.0 # order cost per unit
c0 = 1.5 # fixed order cost
mu = 1.0 # drift of the reference process
sigma = 0.2 # volatility of the reference process
X0 = torch.zeros(num_sample, device=device, dtype=dtype) # initial state of the reference process

#network parameters
width = 250 # width of the hidden layers

#Derived constants
discount_T = np.exp(-r*T)
dt, sqrt_dt = T/N, np.sqrt(T/N)
weights = torch.exp(-r*torch.arange(N, device=device)*dt)
tweights = torch.exp(-r*torch.arange(N, device=device)*dt) * dt

def eval_Z_over_grid(model, X):               # X: (N or N+1, K, d)
    T_, K = X.shape
    Z = model(X.reshape(T_*K,1)).reshape(T_, K)    # (T_, K)
    return Z

def use_he_init(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight)
        nn.init.zeros_(module.bias)

@torch.no_grad()
def inv_cost(X):
    # This computes not only the inventory cost at each stage but also tensorized over the sample paths.
    if X.size(0) != N+1: 
        raise ValueError(f"Sample paths must have {N} time steps")
    else:
        return (tweights @ (h*X[:-1].relu() + p*(-X[:-1]).relu())).unsqueeze(-1) # tensorized total inventory cost over time T for the sample paths.

@torch.no_grad()
def order_cost(dU, c0Bern):
    if dU.size(0) != N:
        raise ValueError(f"We must have {N} order events in each sample path.")
    else:
        return (weights @ (c*dU + c0Bern)).unsqueeze(-1) # tensorized total order cost over time T for the sample paths.

@torch.no_grad()
def sample_generation(X0, device=None, dtype=torch.float32):
    """
    Generate sample paths for 1D reference process X(t) = X0 - mu*t - sigma*W(t)+ sum_j Y_j

    Args:
        num_sample (int): Number of sample paths to generate; this is also the batch size. 
        X0 (torch.Tensor): Initial states for the sample paths. Shape: (num_sample, dim)
        N (int): Number of time intervals to partition the time horizon.
        T (float): Total time horizon.

    Returns:
        samples (torch.Tensor): Generated sample paths. Shape: (num_sample, N, dim)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dt, sqrt_dt = T/N, np.sqrt(T/N) # time step and square root of time step

    Xn = X0.to(device=device, dtype=dtype) # initial state
    #Xn = ensure_batch(X0, num_sample) # initial state for each sample path

    # Generate random increments for the sample paths. 
    # The number of intervals N in first coordinate because we iterate over it. Tensor operations are faster because of how memory is allocated/strides.
    dB = torch.randn(N, num_sample, device=device, dtype=dtype) * sqrt_dt # random Brownian increments
    logZ = torch.randn(N, num_sample, device=device, dtype=dtype)  # random standard normal to generate lognormal.
    Z = torch.exp(ln_mu + ln_sigma * logZ)              # LogNormal with mean 2.0 and variance 1.0

    X = torch.empty(N+1, num_sample, dtype=dtype, device=device) # initialize the sample paths tensor. 
    dU = torch.empty(N, num_sample,dtype=dtype, device=device) # initialize the order sizes tensor. 
    X[0] = Xn # set the initial state for each sample path
    Bern_param = torch.ones(N, num_sample, device=device, dtype=dtype) * lam * dt # Bernoulli distribution parameters. 
    Bern = torch.bernoulli(Bern_param).to(dtype=torch.int32) # Bernoulli distribution. 


    for n in range(N):
        dUn = (Z[n] - Xn).relu() * Bern[n] # order size at time n
        Xn = Xn - mu * dt + sigma * dB[n] + dUn # update the state for each sample path
        dU[n] = dUn # store the order sizes for each sample path
        X[n+1] = Xn # store the state for each sample path

    return X, dB, dU, c0*Bern

def loss_function(Vnet, Znet, X, dU, dB, c0Bern, beta):
    ZN = eval_Z_over_grid(Znet, X[:-1])
    V0 = Vnet(X[0].unsqueeze(-1))
    VN = Vnet(X[-1].unsqueeze(-1))
    #print((((V0 - discount_T*VN + (sigma*weights @ (ZN*dB)).unsqueeze(-1) - inv_cost(X) - order_cost(dU, c0Bern)).relu()) ** 2).shape)
    return torch.mean(-V0 + beta*(((V0 - discount_T*VN + (sigma*weights @ (ZN*dB)).unsqueeze(-1) - inv_cost(X) - order_cost(dU, c0Bern)).relu()) ** 2))

"""
def Vnet(width):
    # value function network
    return nn.Sequential(
        nn.Linear(1, width),
        nn.ELU(),
        nn.Linear(width, width),
        nn.ELU(),
        nn.Linear(width, width),
        nn.ELU(),
        nn.Linear(width, width),
        nn.ELU(),
        nn.Linear(width, 1)
    )

def Znet(width):
    # value function gradient network
    return nn.Sequential(
        nn.Linear(1, width),
        nn.ELU(),
        nn.Linear(width, width),
        nn.ELU(),
        nn.Linear(width, width),
        nn.ELU(),
        nn.Linear(width, width),
        nn.ELU(),
        nn.Linear(width, 1)
    )
"""

vnet_model = Vnet(1, 4, width).to(device=device, dtype=dtype)
znet_model = Znet(1, 4, width).to(device=device, dtype=dtype)
vnet_model.apply(use_he_init)
znet_model.apply(use_he_init)
optimizer = optim.Adam(list(vnet_model.parameters()) + list(znet_model.parameters()), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

beta_m = beta[0]
start_time = time.perf_counter()
for i in range(num_iter):
    X, dB, dU, c0Bern = sample_generation(X0, device, dtype)
    loss = loss_function(vnet_model, znet_model, X, dU, dB, c0Bern, beta_m)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if i % 5000 == 0 and i > 0:
      beta_m = beta[i//5000]
    X0 = X[-1]
    #print(X0)
    if i % 1000 == 0:
        print(f"Iteration {i}, Loss: {loss.item()}, V(0): {vnet_model(torch.zeros(1, device=device, dtype=dtype).unsqueeze(-1)).item()}, Elapsed time: {time.perf_counter() - start_time}")

import matplotlib.pyplot as plt

# --- config ---
xmin, xmax, num = -3.0, 5.0, 1000   # range and resolution

# --- input grid (N,1) because your net expects 1-D inputs ---
xs = torch.linspace(xmin, xmax, num).unsqueeze(-1)  # shape (num, 1)

# (optional) move to same device/dtype as the model
xs = xs.to(next(vnet_model.parameters()).device).to(next(vnet_model.parameters()).dtype)

# --- eval without grads ---
vnet_model.eval()
with torch.no_grad():
    ys = vnet_model(xs).squeeze(-1).detach().cpu().numpy()   # (num,)

# --- plot ---
plt.figure()
plt.plot(xs.squeeze(-1).cpu().numpy(), ys)
plt.xlabel("x")
plt.ylabel("Vmodel(x)")
plt.title("Vmodel over [-3, 5]")
plt.grid(True, alpha=0.3)
plt.show()

# Plot the value function gradient

xs = xs.to(next(znet_model.parameters()).device).to(next(znet_model.parameters()).dtype)

# --- eval without grads ---
znet_model.eval()
with torch.no_grad():
    zs = znet_model(xs).squeeze(-1).detach().cpu().numpy()   # (num,)

# --- plot ---
plt.figure()
plt.plot(xs.squeeze(-1).cpu().numpy(), zs)
plt.xlabel("x")
plt.ylabel("Zmodel(x)")
plt.title("Zmodel over [-3, 5]")
plt.grid(True, alpha=0.3)
plt.show()
