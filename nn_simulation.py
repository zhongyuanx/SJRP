import json
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from networks import Vnet, Znet
import csv
import time
import scipy.stats as stats
import torch.nn.functional as F


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Initialize NN simulator from config and model weights.")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--model-weights", required=True, help="Directory containing vnet_model.pth and znet_model.pth")
    parser.add_argument("--t-max", type=int, default=10_000, help="Max time steps for performance simulation")
    parser.add_argument("--num-samples", type=int, default=10_000, help="Number of Monte Carlo samples")
    parser.add_argument("--lbfgs-max-iter", type=int, default=200, help="LBFGS max iterations for compute_S_solve_inf")
    parser.add_argument("--lbfgs-tol", type=float, default=1e-8, help="LBFGS tolerance for compute_S_solve_inf")
    parser.add_argument("--diffop-eps", type=float, required=True, help="Threshold for diffusion operator")
    return parser.parse_args()

class InventoryPolicySolver(object):
    def __init__(
        self,
        config_path="configurations/1dim/config.json",
        model_weights_dir="configurations/1dim/1d_test/model_weights",
        lbfgs_max_iter=200,
        lbfgs_tol=1e-8,
    ):
        self.config_path = config_path
        self.base_dir = os.path.dirname(os.path.abspath(config_path))
        self.model_weights_dir = model_weights_dir
        self.lbfgs_max_iter = lbfgs_max_iter
        self.lbfgs_tol = lbfgs_tol
        with open(config_path) as f:
            self.config = json.load(f)
        self.net_config = self.config["net_config"]
        self.eqn_config = self.config["eqn_config"]
        self.scaling_factor = self.net_config.get("scaling_factor", 1.0)

        # dimension of the problem; number of items
        self.dim = self.eqn_config["dim"]

        # annual rate of interest
        self.r = self.eqn_config["r"]

        # number of weeks in a year
        self.weeks_per_year = 52

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        # Read the numbers from the csv file
        mu_df = pd.read_csv(self._resolve_path(self.eqn_config["mu_file"]), header=None)
        sigma_df = pd.read_csv(self._resolve_path(self.eqn_config["sigma_file"]), header=None)
        self.mu = torch.tensor(mu_df.iloc[0].values, device=self.device, dtype=torch.float32)
        self.sigma = torch.tensor(sigma_df.iloc[0].values, device=self.device, dtype=torch.float32)

        # Inventory cost rows can have uneven columns, so use csv reader
        rows = []
        with open(self._resolve_path(self.eqn_config["inventory_cost_file"]), newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append([float(x) for x in row if x.strip() != ""])
        self.h = torch.tensor(rows[0], device=self.device, dtype=torch.float32) * self.scaling_factor
        self.p = torch.tensor(rows[1], device=self.device, dtype=torch.float32) * self.scaling_factor

        rows = []
        with open(self._resolve_path(self.eqn_config["ordering_cost_file"]), newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append([float(x) for x in row if x.strip() != ""])
        self.c0 = torch.tensor(rows[0][0], device=self.device, dtype=torch.float32) * self.scaling_factor
        self.c = torch.tensor(rows[1], device=self.device, dtype=torch.float32) * self.scaling_factor

        # network parameters
        self.network_depth = self.net_config["network_depth"]
        self.network_width = self.net_config["network_width"]

        self.r = torch.as_tensor(self.r, device=self.device, dtype=torch.float32)
        self.weeks_per_year = torch.as_tensor(self.weeks_per_year, device=self.device, dtype=torch.float32)


        # Reference process parameters
        S_path = self._resolve_path(self.net_config["S_file"])
        self.S = torch.tensor(pd.read_csv(S_path, header=None).values, dtype=torch.float32).to(self.device)
        # self.S.shape: (1, dim)

        self.vnet_model = Vnet(self.dim, self.network_depth, self.network_width).to(self.device)
        self.znet_model = Znet(self.dim, self.network_depth, self.network_width).to(self.device)

        self._load_models()

    def _resolve_path(self, path_value):
        if os.path.isabs(path_value):
            return path_value
        candidate = os.path.join(self.base_dir, path_value)
        if os.path.exists(candidate):
            return candidate
        repo_root = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(repo_root, path_value)
        if os.path.exists(candidate):
            return candidate
        return path_value

    def _load_models(self):
        vnet_path = self._resolve_path(os.path.join(self.model_weights_dir, "vnet_model.pth"))
        znet_path = self._resolve_path(os.path.join(self.model_weights_dir, "znet_model.pth"))
        self.vnet_model.load_state_dict(torch.load(vnet_path, map_location=self.device))
        self.znet_model.load_state_dict(torch.load(znet_path, map_location=self.device))

    def compute_diffusion_operator(self, x):
        self.vnet_model.eval()
        self.znet_model.eval()
        x0 = torch.as_tensor(x, device=self.device, dtype=torch.float32).clone().requires_grad_(True)
        z_val = self.znet_model(x0)
        z = self.sigma**2 * z_val
        trace = torch.zeros(x0.shape[0], device=self.device)
        for i in range(z.shape[-1]):
            grads = torch.autograd.grad(z[:, i].sum(), x0, retain_graph=True)[0]
            trace += grads[:, i]
        drift = (self.mu * z_val).sum(dim=-1)
        v_val = self.vnet_model(x0).squeeze(-1)
        #print("trace:", trace.mean(), "drift:", drift.mean(), "r*v_val:", (self.r * v_val).mean())
        return -0.5 * trace + drift + self.r * v_val - self.inv_cost(self.h, self.p, x0)

    def inv_cost(self, h, p, x):
        return (h * x.relu() + p * (-x).relu()).sum(dim=-1)

    # Inverse softplus function.
    @staticmethod
    def inv_softplus(x, beta=1.0, threshold=20.0):
        return torch.where(x<=threshold, torch.log(torch.expm1(torch.clamp(beta*x, min=1e-8)))/beta, x)
    
    
    # Solve for the S_opt vector by minimizing the V(S) + c^T S objective.
    def compute_S_solve_inf(self):
        y_init = self.inv_softplus(self.S).to(self.device)
        #y_init = self.S.to(self.device)
        y = y_init.clone().detach().requires_grad_(True)
        # y.shape: (1, dim)
        #print("y:", y)
        optimizer = torch.optim.LBFGS(
            [y],
            max_iter=self.lbfgs_max_iter,
            tolerance_grad=self.lbfgs_tol,
            tolerance_change=self.lbfgs_tol,
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            x = F.softplus(y)
            #x = y
            # x.shape: (1, dim)
            v_val = self.vnet_model(x).squeeze(0).squeeze(-1)
            loss = v_val + torch.sum(self.c * x, dim=-1)
            loss.backward()
            return loss.detach()

        self.vnet_model.eval()
        optimizer.step(closure)
        S_opt = F.softplus(y).detach()
        #S_opt = y.detach()
        return S_opt
    
    # Solve for the S_opt vector by minimizing the (z + c)^T (z + c) objective.
    def compute_S_find_stationary(self):
        y_init = self.inv_softplus(self.S).to(self.device)
        y = y_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS(
            [y],
            max_iter=self.lbfgs_max_iter,
            tolerance_grad=self.lbfgs_tol,
            tolerance_change=self.lbfgs_tol,
        )
        
        def closure():
            optimizer.zero_grad(set_to_none=True)
            x = F.softplus(y)
            # x.shape: (1, dim)
            z_val = self.znet_model(x).squeeze(0).squeeze(-1) # (dim,)
            loss = torch.sum((z_val + self.c) * (z_val + self.c), dim=-1)
            loss.backward()
            return loss.detach()
        
        self.znet_model.eval()
        optimizer.step(closure)
        S_opt = F.softplus(y).detach()
        return S_opt

class NNSimulator(object):
    def __init__(self, policy_solver, diffop_eps=-50.0, seed=42):
        torch.manual_seed(seed)
        self.policy_solver = policy_solver
        self.diffop_eps = diffop_eps
        self.device = policy_solver.device
        self.seed = seed
        self.scaling_factor = policy_solver.scaling_factor
        self.weeks_per_year = policy_solver.weeks_per_year
        self.gamma = torch.exp(-policy_solver.r / self.weeks_per_year)

        self.mu_weekly = policy_solver.mu / self.weeks_per_year
        self.sigma_weekly = policy_solver.sigma / torch.sqrt(self.weeks_per_year)
        self.alpha = self.mu_weekly**2 / self.sigma_weekly**2
        self.theta = self.sigma_weekly**2 / self.mu_weekly
        self.h_weekly = policy_solver.h / self.weeks_per_year
        self.p_weekly = policy_solver.p / self.weeks_per_year

    def load_demand_samples(self, T_max, num_samples):
        # MPS does not implement Gamma sampling; sample on CPU then move to device.
        alpha_cpu = self.alpha.detach().cpu()
        theta_cpu = self.theta.detach().cpu()
        rate_cpu = 1.0 / theta_cpu
        gamma_dist = torch.distributions.Gamma(alpha_cpu, rate_cpu)
        samples = gamma_dist.sample((num_samples, T_max))
        self.demand_samples = samples.to(self.device)
        #self.cum_demand = torch.cumsum(self.demand_samples, dim=1)  # (num_samples, T_max, dim)

    def simulate_performance(self, T_max, num_samples):
        dim = self.policy_solver.dim
        c = self.policy_solver.c
        c0 = self.policy_solver.c0
        gamma = self.gamma

        if not hasattr(self, "demand_samples") or self.demand_samples.shape[:2] != (num_samples, T_max):
            self.load_demand_samples(T_max, num_samples)
        
        #S_opt = self.policy_solver.compute_S_solve_inf()
        S_opt = self.policy_solver.compute_S_find_stationary()

        x = torch.zeros(num_samples, dim, device=self.device)
        total_cost = torch.zeros(num_samples, device=self.device)

        for t in range(T_max):
            total_cost += (gamma ** t) * (self.policy_solver.inv_cost(self.h_weekly, self.p_weekly, x) / self.scaling_factor)

            phi = self.policy_solver.compute_diffusion_operator(x)
            #print("phi:", phi.detach().item(), "t:", t, "x:", x.detach().item())
            order_mask = phi < self.diffop_eps
            if order_mask.any():
                order_size = S_opt - x[order_mask]
                order_cost = c0 + torch.sum(c * order_size, dim=-1)
                total_cost[order_mask] += (gamma ** t) * (order_cost / self.scaling_factor)
                x[order_mask] = S_opt

            x = x - self.demand_samples[:, t, :]

        return total_cost.mean()

# policy = InventoryPolicySolver(
#     config_path="configurations/12dim/config.json",
#     model_weights_dir="12d_mhm_model_weights",
#     lbfgs_max_iter=200,
#     lbfgs_tol=1e-8,
# )
# S_opt = policy.compute_S_solve_inf()
# #S_opt = policy.compute_S_find_stationary()
# print("S_opt:", S_opt)

def main():
    args = _parse_args()
    policy = InventoryPolicySolver(
        config_path=args.config,
        model_weights_dir=args.model_weights,
        lbfgs_max_iter=args.lbfgs_max_iter,
        lbfgs_tol=args.lbfgs_tol,
    )
    #S_opt = policy.compute_S_solve_inf()
    S_opt = policy.compute_S_find_stationary()
    print("S_opt:", S_opt)
    sim = NNSimulator(policy, diffop_eps=args.diffop_eps)
    avg_cost = sim.simulate_performance(T_max=args.t_max, num_samples=args.num_samples)
    print("avg_cost:", float(avg_cost.item()))


if __name__ == "__main__":
    main()
