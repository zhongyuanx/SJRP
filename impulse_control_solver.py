import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from networks import Vnet, Znet
import csv
import time
import os

torch.manual_seed(42)


class ImpulseControlSolver(object):

    def __init__(self, config, run_name=None):
        # Configuration object
        self.config = config
        self.net_config = config.net_config
        self.eqn_config = config.eqn_config
        self.base_dir = getattr(config, "base_dir", "")
        self.run_name = run_name

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # parameter scaling if needed
        self.scaling_factor = self.net_config.scaling_factor

        # PDE dimensions and time parameters
        self.r = self.eqn_config.r
        self.dim = self.eqn_config.dim
        self.time_horizon = self.net_config.time_horizon
        self.num_time_interval = self.net_config.num_time_interval
        self.delta_t = self.time_horizon / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)

        # Derived constants
        #self.discount_T = np.exp(-self.r * self.time_horizon).to(self.device)
        self.discount_T = torch.exp(torch.tensor(-self.r * self.time_horizon, device=self.device))
        self.weights = torch.exp(-self.r * torch.arange(self.num_time_interval, device=self.device) * self.delta_t)
        self.tweights = torch.exp(-self.r * torch.arange(self.num_time_interval, device=self.device) * self.delta_t) * self.delta_t

        # Neural Network architecture parameters
        self.network_width = self.net_config.network_width
        self.network_depth = self.net_config.network_depth
        self.activation = self.net_config.activation
        self.batch_size = self.net_config.batch_size
        self.num_iterations = self.net_config.num_iterations

        # Penalty and learning rate parameters
        self.penalty_schedule = self.net_config.penalty_schedule
        self.penalty_milestones = self.net_config.penalty_milestones
        self.learning_rate_schedule = self.net_config.learning_rate_schedule
        self.learning_rate_milestones = self.net_config.learning_rate_milestones

        # Default save paths if not provided in config
        if not hasattr(self.net_config, "vnet_model_file"):
            self.net_config.vnet_model_file = os.path.join("model_weights", "vnet_model.pth")
        if not hasattr(self.net_config, "znet_model_file"):
            self.net_config.znet_model_file = os.path.join("model_weights", "znet_model.pth")
        if not hasattr(self.net_config, "loss_history_file"):
            self.net_config.loss_history_file = os.path.join("model_weights", "loss_history.csv")

        # If a run_name is provided, force outputs into that run directory.
        if self.run_name:
            run_dir = os.path.join(self.base_dir, self.run_name, "model_weights")
            self.net_config.vnet_model_file = os.path.join(run_dir, os.path.basename(self.net_config.vnet_model_file))
            self.net_config.znet_model_file = os.path.join(run_dir, os.path.basename(self.net_config.znet_model_file))
            self.net_config.loss_history_file = os.path.join(run_dir, os.path.basename(self.net_config.loss_history_file))

        # Reference process hyperparameters
        lam_path = self._resolve_path(self.net_config.lam_file)
        S_path = self._resolve_path(self.net_config.S_file)
        self.lam = torch.tensor(pd.read_csv(lam_path, header=None).values, dtype=torch.float32).to(self.device)
        # self.lam is the rate of ordering events per year.
        self.S = torch.tensor(pd.read_csv(S_path, header=None).values, dtype=torch.float32).to(self.device)
        self.nu = self.net_config.nu
        self.ln_mu = torch.log(self.S**2 / torch.sqrt((1.0 + self.nu**2) * self.S**2))
        self.ln_sigma = torch.sqrt(torch.log(torch.tensor(self.nu**2 + 1.0, device=self.device)))

        # Reference process parameters
        mu_path = self._resolve_path(self.eqn_config.mu_file)
        sigma_path = self._resolve_path(self.eqn_config.sigma_file)
        ordering_cost_path = self._resolve_path(self.eqn_config.ordering_cost_file)
        inventory_cost_path = self._resolve_path(self.eqn_config.inventory_cost_file)
        self.mu = torch.tensor(pd.read_csv(mu_path, header=None).values, dtype=torch.float32).to(self.device)
        self.sigma = torch.tensor(pd.read_csv(sigma_path, header=None).values, dtype=torch.float32).to(self.device)
        costs = list(csv.reader(open(ordering_cost_path)))
        self.fixed_cost = torch.tensor([float(cost) for cost in costs[0]], dtype=torch.float32).to(self.device) * self.scaling_factor
        self.variable_cost = torch.tensor([float(cost) for cost in costs[1]], dtype=torch.float32).to(self.device) * self.scaling_factor
        self.holding_cost = torch.tensor(pd.read_csv(inventory_cost_path, header=None).iloc[0].values, dtype=torch.float32).to(self.device) * self.scaling_factor
        self.backlogging_cost = torch.tensor(pd.read_csv(inventory_cost_path, header=None).iloc[1].values, dtype=torch.float32).to(self.device) * self.scaling_factor

    # def debug_print_params(self):
    #     print("device:", self.device)
    #     print("dim:", self.dim, "r:", self.r, "time_horizon:", self.time_horizon, "num_time_interval:", self.num_time_interval)
    #     print("batch_size:", self.batch_size, "num_iterations:", self.num_iterations)
    #     print("mu:", self.mu.shape, self.mu.detach().cpu().flatten()[:5])
    #     print("sigma:", self.sigma.shape, self.sigma.detach().cpu().flatten()[:5])
    #     print("fixed_cost:", self.fixed_cost.shape, self.fixed_cost.detach().cpu().flatten()[:5])
    #     print("variable_cost:", self.variable_cost.shape, self.variable_cost.detach().cpu().flatten()[:5])
    #     print("holding_cost:", self.holding_cost.shape, self.holding_cost.detach().cpu().flatten()[:5])
    #     print("backlogging_cost:", self.backlogging_cost.shape, self.backlogging_cost.detach().cpu().flatten()[:5])
    #     print("lam:", self.lam.shape, self.lam.detach().cpu().flatten()[:5])
    #     print("S:", self.S.shape, self.S.detach().cpu().flatten()[:5])
    #     print("vnet_model_file:", self.net_config.vnet_model_file)
    #     print("znet_model_file:", self.net_config.znet_model_file)
    #     print("loss_history_file:", self.net_config.loss_history_file)

    def _resolve_path(self, path_value):
        if os.path.isabs(path_value):
            return path_value
        candidate = os.path.join(self.base_dir, path_value)
        if os.path.exists(candidate):
            return candidate
        return path_value

    def sample_generation(self, initial_state, device=None, dtype=torch.float32):
        N = self.num_time_interval
        num_sample = self.batch_size
        device = device or self.device
        dB = torch.randn(N, num_sample, self.dim, device=device, dtype=dtype) * self.sqrt_delta_t
        logZ = torch.randn(N, num_sample, self.dim, device=device, dtype=dtype)
        Z = torch.exp(self.ln_mu + self.ln_sigma * logZ)

        X = torch.empty(N + 1, num_sample, self.dim, dtype=dtype, device=device)
        dU = torch.empty(N, num_sample, self.dim, dtype=dtype, device=device)
        X[0] = initial_state.to(device=device, dtype=dtype)
        Bern_param = torch.ones(N, num_sample, device=device, dtype=dtype) * self.lam * self.delta_t
        Bern = torch.bernoulli(Bern_param).to(dtype=torch.int32)

        for n in range(N):
            dU[n] = (Z[n] - X[n]).relu() * Bern[n][:, None]
            X[n + 1] = X[n] - self.mu * self.delta_t + self.sigma * dB[n] + dU[n]

        return X, dB, dU, Bern

    @staticmethod
    def eval_Z_over_grid(model, X):
        T_, K, d = X.shape
        Z = model(X.reshape(T_ * K, d)).reshape(T_, K, d)
        return Z

    @staticmethod
    def use_he_init(module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def inv_cost(self, X):
        if X.size(0) != self.num_time_interval + 1:
            raise ValueError(f"Sample paths must have {self.num_time_interval} time steps")
        return (self.tweights[:, None] * (self.holding_cost * X[:-1].relu() + self.backlogging_cost * (-X[:-1]).relu()).sum(dim=-1)).sum(dim=0)

    def order_cost(self, dU, Bern):
        if dU.size(0) != self.num_time_interval:
            raise ValueError(f"We must have {self.num_time_interval} order events in each sample path.")
        return (self.weights[:, None] * (self.fixed_cost * Bern + (self.variable_cost * dU).sum(dim=-1))).sum(dim=0)

    def loss_function(self, Vnet, Znet, X, dU, dB, Bern, beta):
        ZN = self.eval_Z_over_grid(Znet, X[:-1])
        V0 = Vnet(X[0]).squeeze(-1) 
        VN = Vnet(X[-1]).squeeze(-1)
        return torch.mean(-V0 + beta * (((V0 - self.discount_T * VN + (self.weights[:, None] * (self.sigma * ZN * dB).sum(dim=-1)).sum(dim=0) - self.inv_cost(X) - self.order_cost(dU, Bern)).relu()) ** 2))

    def train(self):
        vnet_model = Vnet(self.dim, self.network_depth, self.network_width).to(self.device)
        znet_model = Znet(self.dim, self.network_depth, self.network_width).to(self.device)
        vnet_model.apply(self.use_he_init)
        znet_model.apply(self.use_he_init)

        learning_rate_schedule = torch.tensor(self.learning_rate_schedule, device=self.device)
        learning_rate_milestones = torch.tensor(self.learning_rate_milestones, device=self.device)
        learning_rate = learning_rate_schedule[0]
        penalty_schedule = torch.tensor(self.penalty_schedule, device=self.device)
        penalty_milestones = torch.tensor(self.penalty_milestones, device=self.device)
        penalty = penalty_schedule[0]

        optimizer = optim.Adam(list(vnet_model.parameters()) + list(znet_model.parameters()), lr=learning_rate)
        X0 = torch.zeros(self.batch_size, self.dim, device=self.device, dtype=torch.float32)
        loss_history = []
        start_time = time.perf_counter()
        for iteration in range(self.num_iterations):
            X, dB, dU, Bern = self.sample_generation(X0)
            loss = self.loss_function(vnet_model, znet_model, X, dU, dB, Bern, penalty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if torch.any(learning_rate_milestones == iteration):
                learning_rate = learning_rate_schedule[torch.where(learning_rate_milestones == iteration)].item()
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rate
            #if iteration in penalty_milestones:
            #    penalty = penalty_schedule[penalty_milestones.index(iteration)]
            if torch.any(penalty_milestones == iteration):
                penalty = penalty_schedule[torch.where(penalty_milestones == iteration)].item()
            X0 = X[-1]
            if iteration % 1000 == 0:
                v0_input = torch.zeros(1, self.dim, device=self.device, dtype=torch.float32)
                v0_val = vnet_model(v0_input).squeeze(-1).item()
                print(f"Iteration {iteration}, Loss: {loss.item()}, V(0): {v0_val}, Elapsed time: {time.perf_counter() - start_time}")
            loss_history.append([iteration, loss.item()])
        self.vnet_model = vnet_model
        self.znet_model = znet_model
        self.save_model(self.vnet_model, self.znet_model)
        self.save_loss_history(loss_history)
        

    def save_model(self, vnet_model, znet_model):
        os.makedirs(os.path.dirname(self.net_config.vnet_model_file), exist_ok=True)
        os.makedirs(os.path.dirname(self.net_config.znet_model_file), exist_ok=True)
        torch.save(vnet_model.state_dict(), self.net_config.vnet_model_file)
        torch.save(znet_model.state_dict(), self.net_config.znet_model_file)

    def load_model(self, vnet_model, znet_model):
        vnet_model.load_state_dict(torch.load(self.net_config.vnet_model_file))
        znet_model.load_state_dict(torch.load(self.net_config.znet_model_file))

    def save_loss_history(self, loss_history):
        os.makedirs(os.path.dirname(self.net_config.loss_history_file), exist_ok=True)
        with open(self.net_config.loss_history_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Iteration', 'Loss'])
            writer.writerows(loss_history)
