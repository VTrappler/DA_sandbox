#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from solvers import RK4_step, integrate_step
from dynmodels import DynamicalModel as Model


class Lorenz93Model(Model):
    dim = 40
    solver = RK4_step
    dt = 0.02

    @classmethod
    def set_dim(cls, dim):
        cls.dim = dim

    @classmethod
    def dotfunction(cls, t, x):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        d = np.zeros(cls.dim)
        for i in range(cls.dim):
            d[i] = (x[(i + 1) % cls.dim] - x[i - 2]) * x[i - 1] - x[i] + 8
        return d

    @classmethod
    def integrate(cls, t0, x0, Nsteps):
        return integrate_step(cls.solver, cls.dotfunction, t0, x0, cls.dt, Nsteps)


def main():
    Ndim = 200
    Lorenz93Model.dim = Ndim
    lorenz40 = Lorenz93Model()
    x0 = np.random.normal(0, 1, lorenz40.dim)
    lorenz40.set_initial_state(0, x0)
    lorenz40.forward(5000)
    plt.imshow(lorenz40.state_vector, aspect="auto")
    plt.show()


if __name__ == "__main__":
    main()
