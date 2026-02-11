import numpy as np
from scipy.optimize import root_scalar

class Sulem_sS_Solver:
    def __init__(self, r, p, h, c, c0, mu, sigma):
        self.r = float(r)
        self.p = np.asarray(p, dtype=float)
        self.h = np.asarray(h, dtype=float)
        self.c = np.asarray(c, dtype=float)
        self.c0 = np.asarray(c0, dtype=float)
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        try:
            self.p, self.h, self.c, self.c0, self.mu, self.sigma = np.broadcast_arrays(
                self.p, self.h, self.c, self.c0, self.mu, self.sigma
            )
        except ValueError as exc:
            raise ValueError("p, h, c, c0, mu, sigma must be broadcastable to the same shape") from exc
        self.shape = self.p.shape
        self.lambda1, self.lambda2 = self._lambdas()

    def _lambdas(self):
        lambda1 = (np.sqrt(self.mu**2 + 2 * self.r * self.sigma**2) - self.mu) / self.sigma**2
        lambda2 = (-np.sqrt(self.mu**2 + 2 * self.r * self.sigma**2) - self.mu) / self.sigma**2
        return lambda1, lambda2

    def _params_at(self, idx):
        return (
            self.p[idx],
            self.h[idx],
            self.c[idx],
            self.c0[idx],
            self.lambda1[idx],
            self.lambda2[idx],
        )

    def _M(self, x, idx):
        p, h, c, c0, lambda1, lambda2 = self._params_at(idx)
        return lambda1 * (
            (-p + c * self.r) * x
            + (p + h) * (1 - np.exp(-lambda2 * x)) / lambda2
            - c0 * self.r
        ) / (h + c * self.r)

    def _N(self, x, idx):
        p, h, c, _, lambda1, lambda2 = self._params_at(idx)
        return (
            (-p + c * self.r) * np.exp(lambda1 * x)
            + (p + h)
            / (lambda1 - lambda2)
            * (lambda1 * np.exp((lambda1 - lambda2) * x) - lambda2)
        ) / (h + c * self.r)

    def _s_eqn_idx(self, s, idx):
        return np.log((np.exp(self._M(s, idx)) - self._N(s, idx)) + 1.0)

    def solve(self, initial_guess=-3.0):
        initial_guess = np.asarray(initial_guess, dtype=float)
        if self.shape == ():
            sol = root_scalar(lambda s: self._s_eqn_idx(s, ()), x0=float(initial_guess))
            s = sol.root
            S = self._M(s, ()) / self.lambda1
            return s, S, sol

        initial_guess = np.broadcast_to(initial_guess, self.shape)
        s_out = np.empty(self.shape, dtype=float)
        S_out = np.empty(self.shape, dtype=float)
        sol_out = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            sol = root_scalar(lambda s: self._s_eqn_idx(s, idx), x0=float(initial_guess[idx]))
            s = sol.root
            S = self._M(s, idx) / self.lambda1[idx]
            s_out[idx] = s
            S_out[idx] = S
            sol_out[idx] = sol
        return s_out, S_out, sol_out


def main():
    # Example: vector parameters (same shape)
    # p = np.array([100.0, 120.0, 80.0])
    # h = np.array([2.0, 2.5, 1.5])
    # c = np.array([0.1, 0.12, 0.08])
    # mu = np.array([40.0, 50.0, 35.0])
    # sigma = np.array([20.0, 25.0, 18.0])

    p = 2.0
    h = 0.5
    c = 1.0
    mu = 1.0
    sigma = 0.2

    solver = Sulem_sS_Solver(
        r=0.05,
        p=p,
        h=h,
        c=c,
        c0=1.5,
        mu=mu,
        sigma=sigma,
    )
    s, S, sol = solver.solve(initial_guess=0.0)
    print(p, h)
    print("s:", s)
    print("S:", S)
    if solver.shape == ():
        print("Converged:", sol.converged)
    else:
        converged = np.array([sol[idx].converged for idx in np.ndindex(solver.shape)])
        print("Converged:", converged.reshape(solver.shape))


if __name__ == "__main__":
    main()
