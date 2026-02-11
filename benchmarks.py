import csv
import math
import os
import time

import numpy as np
import pandas as pd
import scipy.stats as stats

from Sulem_sS_vectorized import Sulem_sS_Solver


class BenchmarkPolicies:
    def __init__(self, config_path="configurations/12dim/config.json", seed=42):
        self.config_path = config_path
        self.base_dir = os.path.dirname(os.path.abspath(config_path))
        self.config = self._load_config(config_path)
        np.random.seed(seed)

        self.dim = self.config["eqn_config"]["dim"]
        self.r = self.config["eqn_config"]["r"]
        self.weeks_per_year = 52
        self.gamma = np.exp(-self.r / self.weeks_per_year)

        # Resolve input paths
        self.mu_path = self._resolve_path(self.config["eqn_config"]["mu_file"])
        self.sigma_path = self._resolve_path(self.config["eqn_config"]["sigma_file"])
        self.inventory_cost_path = self._resolve_path(self.config["eqn_config"]["inventory_cost_file"])
        self.ordering_cost_path = self._resolve_path(self.config["eqn_config"]["ordering_cost_file"])

        self.rs_r_path = self._output_path(self.config["net_config"].get("ordering_frequency_file", "RS_R.csv"))
        self.rs_cost_path = self._output_path("RS_cost.csv")
        self.rs_s_path = self._output_path("RS_S.csv")

        self.qs_q_path = self._output_path("QS_Q.csv")
        self.qs_cost_path = self._output_path("QS_cost.csv")
        self.qs_s_path = self._output_path(self.config["net_config"].get("order_up_to_vector_file", "QS_S.csv"))

        self.can_cost_path = self._output_path("can_order_cost.csv")
        self.can_s_path = self._output_path("can_order_s_lower.csv")
        self.can_S_path = self._output_path("can_order_S_upper.csv")
        self.can_o_path = self._output_path("can_order_o.csv")
        self.can_freq_path = self._output_path("can_order_order_frequency.csv")

        # Load numeric inputs
        mu_df = pd.read_csv(self.mu_path, header=None)
        self.mu = mu_df.iloc[0].astype(np.float64).to_numpy()
        self.mu_weekly = self.mu / self.weeks_per_year

        sigma_df = pd.read_csv(self.sigma_path, header=None)
        self.sigma = sigma_df.iloc[0].astype(np.float64).to_numpy()
        self.sigma_weekly = self.sigma / np.sqrt(self.weeks_per_year)

        inventory_cost_df = pd.read_csv(self.inventory_cost_path, header=None)
        self.h = inventory_cost_df.iloc[0].astype(np.float64).to_numpy()
        self.p = inventory_cost_df.iloc[1].astype(np.float64).to_numpy()
        self.h_weekly = self.h / self.weeks_per_year
        self.p_weekly = self.p / self.weeks_per_year

        rows = []
        with open(self.ordering_cost_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append([float(x) for x in row if x.strip() != ""])
        self.c0 = rows[0][0]
        self.c = np.array(rows[1])

        # Gamma params for weekly demand
        self.alpha = self.mu_weekly**2 / self.sigma_weekly**2
        self.theta = self.sigma_weekly**2 / self.mu_weekly

    def _load_config(self, config_path):
        import json
        with open(config_path) as f:
            return json.load(f)

    def _resolve_path(self, path_value):
        if os.path.isabs(path_value):
            return path_value
        candidate = os.path.join(self.base_dir, path_value)
        if os.path.exists(candidate):
            return candidate
        return path_value

    def _output_path(self, path_value):
        if os.path.isabs(path_value):
            return path_value
        return os.path.join(self.base_dir, path_value)

    # ---------- RS ----------
    def _first_order_R(self, R, y, c_i, h_i, p_i, alpha_i, theta_i):
        func = (1 - self.gamma**R) * c_i + np.sum(np.array([
            self.gamma**r * (
                h_i * stats.gamma.cdf(y, (r+1) * alpha_i, loc=0, scale=theta_i)
                - p_i * (1 - stats.gamma.cdf(y, (r+1) * alpha_i, loc=0, scale=theta_i))
            )
            for r in range(R)
        ]))
        grad = np.sum(np.array([
            self.gamma**r * (h_i + p_i) * stats.gamma.pdf(y, (r+1) * alpha_i, loc=0, scale=theta_i)
            for r in range(R)
        ]))
        return func, grad

    def _newton_raphson_R(self, R, x0, c_i, h_i, p_i, alpha_i, theta_i, tol=1e-10, max_iter=1000):
        x = x0
        for _ in range(max_iter):
            func, grad = self._first_order_R(R, x, c_i, h_i, p_i, alpha_i, theta_i)
            if abs(func) < tol:
                return x
            x = x - func / grad
        return x

    def _optimal_S(self, R):
        S = np.ones(self.dim)
        for i in range(self.dim):
            S[i] = self._newton_raphson_R(R, S[i], self.c[i], self.h[i], self.p[i], self.alpha[i], self.theta[i])
        return S

    def run_rs(self, R_min=1, R_max=100, num_samples=1_000_000):
        start_time = time.time()
        demand_samples = np.random.gamma(shape=self.alpha, scale=self.theta, size=(num_samples, R_max, self.dim))
        cum_demand = np.cumsum(demand_samples, axis=1)

        def optimal_cost(R):
            S = self._optimal_S(R)
            avg_cost = self.c0 * (1 + self.gamma**R / (1 - self.gamma**R)) if np.any(S) else self.c0 * self.gamma**R / (1 - self.gamma**R)
            cum_demand_R = cum_demand[:, :R, :]
            inv_cost_matrix = np.maximum(self.h_weekly * (S - cum_demand_R), -self.p_weekly * (S - cum_demand_R))
            avg_inv_cost = np.sum(self.gamma ** np.arange(R) * np.sum(inv_cost_matrix, axis=-1)) / (1 - self.gamma**R) / num_samples
            avg_ord_cost = np.dot(self.c, S + self.mu_weekly * R / (1 - self.gamma**R))
            return avg_cost + avg_inv_cost + avg_ord_cost, S

        left, right = R_min, R_max
        for _ in range(100):
            if right - left <= 1:
                break
            m1 = int(math.ceil(left + (right - left) / 3.0))
            m2 = int(right - (right - left) / 3.0)
            f1 = optimal_cost(m1)[0]
            f2 = optimal_cost(m2)[0]
            if f1 < f2:
                right = m2
            else:
                left = m1
        f_left = optimal_cost(left)
        f_right = optimal_cost(right)
        R_opt, cost_opt = (left, f_left) if f_left[0] < f_right[0] else (right, f_right)

        min_cost, S_opt = cost_opt[0], cost_opt[1]
        with open(self.rs_r_path, "w", newline="") as f:
            csv.writer(f).writerow([R_opt])
        with open(self.rs_cost_path, "w", newline="") as f:
            csv.writer(f).writerow([min_cost])
        with open(self.rs_s_path, "w", newline="") as f:
            csv.writer(f).writerow(S_opt)
        elapsed = time.time() - start_time
        print(f"run_rs time: {elapsed:.2f}s")
        return R_opt, min_cost, S_opt, elapsed

    # ---------- QS ----------
    def run_qs(self, R_max=200, num_samples=1_000_000):
        start_time = time.time()
        R_opt = float(np.loadtxt(self.rs_r_path, delimiter=","))
        demand_samples = np.random.gamma(shape=self.alpha, scale=self.theta, size=(num_samples, R_max, self.dim))
        cum_demand = np.cumsum(demand_samples, axis=1)
        cum_demand_all = np.sum(cum_demand, axis=-1)

        def R_Q(Q):
            bool_matrix = (cum_demand_all < Q)
            RQ = bool_matrix.sum(axis=-1)
            true_column = np.full((num_samples, 1), True, dtype=bool)
            bool_matrix = np.concatenate((true_column, bool_matrix), axis=1)[:, :R_max]
            discount = self.gamma ** np.arange(R_max)
            discount = discount[None, :] * bool_matrix
            return RQ, discount

        def first_order_Q(Q):
            RQ, discount = R_Q(Q)
            def func_and_grad(S):
                func = np.zeros(self.dim)
                grad = np.zeros(self.dim)
                for i in range(self.dim):
                    Di = cum_demand[:, :, i]
                    func[i] = np.average(np.sum(discount * ((self.h_weekly[i] + self.p_weekly[i]) * (S[i] > Di) - self.p_weekly[i]), axis=-1))
                    rem_inv = np.concatenate((S[i] * np.ones((num_samples, 1)), S[i] - Di), axis=1)[:, :R_max]
                    grad[i] = np.average(np.sum(discount * ((self.h_weekly[i] + self.p_weekly[i]) * stats.gamma.pdf(rem_inv, a=self.alpha[i], scale=self.theta[i])), axis=-1))
                func = func + np.average(1 - self.gamma**RQ) * self.c
                return func, grad
            return func_and_grad

        def newton_raphson(Q, x0, tol=1e-2, max_iter=1000):
            x = x0
            active = np.full(self.dim, True, dtype=bool)
            f_and_g = first_order_Q(Q)
            for _ in range(max_iter):
                if not active.any():
                    break
                func, grad = f_and_g(x)
                for i in range(self.dim):
                    if abs(func[i]) < tol:
                        active[i] = False
                    else:
                        x[i] = x[i] - func[i] / grad[i]
            return x

        def inv_cost(x):
            return np.maximum(self.h_weekly * x, -self.p_weekly * x)

        def optimal_cost(Q):
            S = newton_raphson(Q, R_opt * np.ones(self.dim))
            RQ, discount = R_Q(Q)
            norm_factor = np.average(1 - self.gamma**RQ)
            total_cost = self.c0 / norm_factor + np.dot(self.c, S)
            inv_cost_matrix = inv_cost(S - cum_demand)
            total_cost += np.sum(discount * np.sum(inv_cost_matrix, axis=-1)) / num_samples / norm_factor
            for i in range(self.dim):
                Di = cum_demand[:, :, i]
                rows = np.arange(Di.shape[0])
                Di_RQ = Di[rows, RQ]
                total_cost += self.c[i] * np.average((self.gamma**RQ) * Di_RQ) / norm_factor
            return total_cost, S

        Q_lb = sum(self.mu_weekly) * np.max(R_opt - 5, 0)
        Q_ub = sum(self.mu_weekly) * (R_opt + 5)

        left, right = Q_lb, Q_ub
        for _ in range(100):
            if right - left < 1e-2 * left:
                break
            m1 = left + (right - left) / 3.0
            m2 = right - (right - left) / 3.0
            f1 = optimal_cost(m1)[0]
            f2 = optimal_cost(m2)[0]
            if f1 < f2:
                right = m2
            else:
                left = m1
        Q_opt = 0.5 * (left + right)
        cost_opt = optimal_cost(Q_opt)
        min_cost, S_opt = cost_opt[0], cost_opt[1]

        with open(self.qs_q_path, "w", newline="") as f:
            csv.writer(f).writerow([Q_opt])
        with open(self.qs_cost_path, "w", newline="") as f:
            csv.writer(f).writerow([min_cost])
        with open(self.qs_s_path, "w", newline="") as f:
            csv.writer(f).writerow(S_opt)
        elapsed = time.time() - start_time
        print(f"run_qs time: {elapsed:.2f}s")
        return Q_opt, min_cost, S_opt, elapsed

    # ---------- Can-order ----------
    def run_can_order(self):
        start_time = time.time()
        kappa1_array = np.arange(0.05, 1.01, 0.05)
        kappa2_array = np.arange(0, 1.01, 0.1)

        s_array = np.zeros((len(kappa1_array), self.dim))
        S_array = np.zeros((len(kappa1_array), self.dim))
        for k, kappa1 in enumerate(kappa1_array):
            solver = Sulem_sS_Solver(
                r=self.r,
                p=self.p,
                h=self.h,
                c=self.c,
                c0=kappa1 * self.c0,
                mu=self.mu,
                sigma=self.sigma,
            )
            s, S, _ = solver.solve(initial_guess=0.0)
            s_array[k, :] = s
            S_array[k, :] = S

        T_max = 10_000
        num_samples = 10_000
        demand_samples = np.random.gamma(shape=self.alpha, scale=self.theta, size=(num_samples, T_max, self.dim))

        def inv_cost(x):
            return np.maximum(self.h_weekly * x, -self.p_weekly * x)

        def performance_simulation(s, o, S):
            total_cost = np.zeros(num_samples)
            order_times = np.zeros((num_samples, 2))
            inventory = np.zeros((num_samples, T_max + 1, self.dim))
            for t in range(T_max):
                order = (inventory[:, t, :] <= s).any(axis=-1)
                order_times[:, 0] = np.maximum(order_times[:, 0], order * t)
                order_times[:, 1] += order
                total_cost += (self.gamma**t) * self.c0 * order
                order_size = order.reshape(-1, 1) * (inventory[:, t, :] <= o) * (S - inventory[:, t, :])
                total_cost += (self.gamma**t) * np.sum(self.c * order_size, axis=-1)
                inventory[:, t + 1, :] = inventory[:, t, :] + order_size
                inventory[:, t + 1, :] -= demand_samples[:, t, :]
                total_cost += (self.gamma**t) * inv_cost(inventory[:, t + 1, :]).sum(axis=-1)
            total_cost = np.average(total_cost)
            order_duration = np.sum(order_times[:, 0]) / np.sum(order_times[:, 1])
            return total_cost, self.weeks_per_year / order_duration

        total_cost_array = np.zeros((len(kappa1_array), len(kappa2_array)))
        order_frequency_array = np.zeros((len(kappa1_array), len(kappa2_array)))
        for k1 in range(len(kappa1_array)):
            for k2, kappa2 in enumerate(kappa2_array):
                o = kappa2 * s_array[k1, :] + (1 - kappa2) * S_array[k1, :]
                total_cost_array[k1, k2], order_frequency_array[k1, k2] = performance_simulation(
                    s_array[k1, :], o, S_array[k1, :]
                )

        total_cost_array[np.isnan(total_cost_array)] = np.inf
        min_index_flattened = np.argmin(total_cost_array)
        row, col = np.unravel_index(min_index_flattened, total_cost_array.shape)

        with open(self.can_cost_path, "w", newline="") as f:
            csv.writer(f).writerow([total_cost_array[row, col]])
        with open(self.can_s_path, "w", newline="") as f:
            csv.writer(f).writerow(s_array[row, :])
        with open(self.can_S_path, "w", newline="") as f:
            csv.writer(f).writerow(S_array[row, :])
        with open(self.can_o_path, "w", newline="") as f:
            csv.writer(f).writerow(kappa2_array[col] * s_array[row, :] + (1 - kappa2_array[col]) * S_array[row, :])
        with open(self.can_freq_path, "w", newline="") as f:
            csv.writer(f).writerow([order_frequency_array[row, col]])

        end_time = time.time()
        print(f"run_can_order time: {end_time - start_time:.2f}s")
        return {
            "row": row,
            "col": col,
            "min_cost": total_cost_array[row, col],
            "order_frequency": order_frequency_array[row, col],
            "kappa1": kappa1_array[row],
            "kappa2": kappa2_array[col],
            "time_sec": end_time - start_time,
        }


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run benchmark policies (RS/QS/can-order).")
    parser.add_argument("--config", default="configurations/12dim/config.json", help="Path to config.json")
    parser.add_argument(
        "--run",
        choices=["rs", "qs", "can_order", "all"],
        default="all",
        help="Which benchmark to run",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    bench = BenchmarkPolicies(args.config)
    if args.run in ("rs", "all"):
        bench.run_rs()
    if args.run in ("qs", "all"):
        bench.run_qs()
    if args.run in ("can_order", "all"):
        bench.run_can_order()


if __name__ == "__main__":
    main()
