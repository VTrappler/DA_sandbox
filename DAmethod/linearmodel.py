#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from solvers.solvers import integrate_step
from DAmethod.DAModel import DAModel


class LinearModel(DAModel):
    """Linear model of the form
    x_{k+1} = Mx_k + w_k
    y_k = Hx_k + v_k
    """

    dt = 1

    def __init__(self, M, H, Q, R):
        self.M = M
        self.H = H
        self.Q = Q
        self.R = R

    def apply_model(self, t, x):
        """Linear model"""
        return self.M @ x

    def apply_observation(self, t, x):
        return self.H @ x

    def step(self, t, x):
        return self.apply_model(t, x) + np.random.multivariate_normal(
            np.ones(len(self.Q)), cov=self.Q
        )

    @classmethod
    def integrate(cls, t0, x0, Nsteps):
        return integrate_step(cls.solver, cls.dotfunction, t0, x0, cls.dt, Nsteps)


def main():
    dim = 2
    x0 = np.random.normal(0, 1, size=dim).reshape(dim, 1)
    M = np.random.randint(10, size=dim ** 2).reshape(dim, dim)
    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1, 0], [0, 1]])
    H = np.array([[1, 0]])
    lm = LinearModel(M, H, Q, R)
    print(lm.step(0, x0))


if __name__ == "__main__":
    main()
