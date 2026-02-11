import argparse
import csv
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from networks import Vnet, Znet


def _resolve_path(base_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    candidate = os.path.join(base_dir, path_value)
    if os.path.exists(candidate):
        return candidate
    return path_value


def _load_vector(path_value):
    df = pd.read_csv(path_value, header=None)
    return df.iloc[0].astype(np.float64).to_numpy()


def _parse_args():
    parser = argparse.ArgumentParser(description="LBFGS solve for x >= 0 minimizing Vnet(x) + c^T x.")
    parser.add_argument("--config", default="configurations/1dim/config.json", help="Path to config.json")
    parser.add_argument("--model-weights", default="model_weights", help="Directory with vnet_model.pth and znet_model.pth")
    parser.add_argument("--c-csv", default=None, help="Optional CSV path for vector c (single row).")
    parser.add_argument("--max-iter", type=int, default=200, help="LBFGS max iterations")
    parser.add_argument("--tol", type=float, default=1e-9, help="LBFGS tolerance")
    parser.add_argument("--plot", action="store_true", help="Plot Vnet(x) + c^T x for 1D")
    parser.add_argument("--plot-y", action="store_true", help="Plot Vnet(softplus(y)) + c^T softplus(y) for 1D")
    parser.add_argument("--save-fig", default=None, help="Optional path to save the plot instead of showing it")
    parser.add_argument("--x-min", type=float, default=0.0, help="Min x for plot (1D)")
    parser.add_argument("--x-max", type=float, default=10.0, help="Max x for plot (1D)")
    parser.add_argument("--x-steps", type=int, default=200, help="Number of points for plot (1D)")
    parser.add_argument("--x0", type=float, default=0.0, help="Initial x for LBFGS (1D only)")
    return parser.parse_args()


def main():
    args = _parse_args()
    with open(args.config) as f:
        config = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(args.config))

    dim = config["eqn_config"]["dim"]
    network_depth = config["net_config"]["network_depth"]
    network_width = config["net_config"]["network_width"]

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    vnet_model = Vnet(dim, network_depth, network_width).to(device)
    znet_model = Znet(dim, network_depth, network_width).to(device)

    vnet_path = _resolve_path(base_dir, os.path.join(args.model_weights, "vnet_model.pth"))
    znet_path = _resolve_path(base_dir, os.path.join(args.model_weights, "znet_model.pth"))
    vnet_model.load_state_dict(torch.load(vnet_path, map_location=device))
    znet_model.load_state_dict(torch.load(znet_path, map_location=device))

    if args.c_csv is None:
        # Use the ordering cost vector from config (same as nn_simulation.py self.c)
        ordering_cost_path = _resolve_path(base_dir, config["eqn_config"]["ordering_cost_file"])
        rows = []
        with open(ordering_cost_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append([float(x) for x in row if x.strip() != ""])
        c_np = np.array(rows[1], dtype=np.float64)
    else:
        c_np = _load_vector(_resolve_path(base_dir, args.c_csv))
    if c_np.shape[0] != dim:
        raise ValueError(f"c has dim {c_np.shape[0]} but expected {dim}")
    c = torch.as_tensor(c_np, device=device, dtype=torch.float32)

    # Unconstrained variable; x = softplus(y) enforces nonnegativity
    if dim == 1:
        x0 = torch.tensor([args.x0], device=device, dtype=torch.float32)
        y_init = torch.log(torch.expm1(torch.clamp(x0, min=1e-8)))
    else:
        y_init = torch.zeros(dim, device=device, dtype=torch.float32)
    y = y_init.clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS([y], max_iter=args.max_iter, tolerance_grad=args.tol, tolerance_change=args.tol)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        x = F.softplus(y)
        v_val = vnet_model(x.unsqueeze(0)).squeeze(0).squeeze(-1)
        loss = v_val + torch.dot(c, x)
        loss.backward()
        return loss

    vnet_model.eval()
    znet_model.eval()
    optimizer.step(closure)
    x_opt = F.softplus(y).detach().cpu().numpy()
    print("x_opt:", x_opt)

    if args.plot or args.plot_y:
        if dim != 1:
            raise ValueError("--plot/--plot-y only supports dim=1")
        if args.plot_y:
            ys = torch.linspace(args.x_min, args.x_max, args.x_steps, device=device).unsqueeze(-1)
            xs = F.softplus(ys)
            with torch.no_grad():
                v_vals = vnet_model(xs).squeeze(-1)
                obj_vals = v_vals + c[0] * xs.squeeze(-1)
                min_idx = torch.argmin(obj_vals)
                print("grid_min_y:", float(ys[min_idx].item()), "grid_min_val:", float(obj_vals[min_idx].item()))
            plt.plot(ys.detach().cpu().numpy(), obj_vals.detach().cpu().numpy())
            plt.xlabel("y")
            plt.ylabel("Vnet(softplus(y)) + c^T softplus(y)")
            plt.title("Objective vs y")
        else:
            xs = torch.linspace(args.x_min, args.x_max, args.x_steps, device=device).unsqueeze(-1)
            with torch.no_grad():
                v_vals = vnet_model(xs).squeeze(-1)
                obj_vals = v_vals + c[0] * xs.squeeze(-1)
                min_idx = torch.argmin(obj_vals)
                print("grid_min_x:", float(xs[min_idx].item()), "grid_min_val:", float(obj_vals[min_idx].item()))
            plt.plot(xs.detach().cpu().numpy(), obj_vals.detach().cpu().numpy())
            plt.xlabel("x")
            plt.ylabel("Vnet(x) + c^T x")
            plt.title("Objective vs x")
        plt.tight_layout()
        if args.save_fig:
            plt.savefig(args.save_fig, dpi=150)
            print(f"Saved plot to {args.save_fig}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
