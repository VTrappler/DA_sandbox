#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from solvers.solvers import RK4_step, integrate_step
import matplotlib.pyplot as plt
from dynamicalsystems.dynmodels import DynamicalModel as Model


class Lorenz63Model(Model):
    dim = 3
    solver = RK4_step

    @classmethod
    def dotfunction(cls, t, x):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        d = np.zeros(3)
        d[0] = 10 * (x[1] - x[0])
        d[1] = 28 * x[0] - x[1] - x[0] * x[2]
        d[2] = -x[2] * 8.0 / 3.0 + x[1] * x[0]
        return d

    @classmethod
    def integrate(cls, t0, x0, Nsteps):
        return integrate_step(cls.solver, cls.dotfunction, t0, x0, cls.dt, Nsteps)

    @classmethod
    def TLM(self, x):
        return np.array(
            [
                [-10, 10, 0],
                [28 - x[2], -1 - x[0]],
                [x[1], x[0], -8.0 / 3.0],
            ]
        )


def main():
    lorenz63 = Lorenz63Model()
    dt = 0.02
    Lorenz63Model.dt = dt
    lorenz63.set_initial_state(0, np.array([0, 1, 0]))
    lorenz63.forward(5000)
    fig = plt.figure()

    for i in range(3):
        ax = fig.add_subplot(3, 2, 2 * i + 1)
        ax.plot(lorenz63.state_vector[i, :])

    ax3d = fig.add_subplot(3, 2, (2, 4), projection="3d")
    ax3d.scatter(*lorenz63.state_vector, c=lorenz63.t)
    plt.show()


if __name__ == "__main__":
    main()
