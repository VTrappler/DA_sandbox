# -*- coding: utf-8 -*-
#!/usr/bin/env python


import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import scipy


class LorenzModel93:
    def __init__(self, dimension=40):
        self.state = np.empty((dimension, 1))
        self.history = np.empty(0), np.empty((dimension, 0))
        self.dim = dimension

    def initial_state(self, init):
        self.initstate = init

    def L96(self, x, t):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        d = np.zeros(self.dim)
        for i in range(self.dim):
            d[i] = (x[(i + 1) % self.dim] - x[i - 2]) * x[i - 1] - x[i] + 8
        return d

    def integrate(self, t):
        integration = scipy.integrate.odeint(func=self.L96, y0=self.initstate, t=t)
        self.history = np.concatenate((self.history[0], t)), np.concatenate(
            (self.history[1], integration.T), axis=1
        )

    def step(self, dt, Nsteps):
        try:
            self.integrate(
                np.arange(
                    self.history[0][-1],
                    self.history[0][-1] + Nsteps * dt,
                    dt,
                )
            )
        except IndexError:
            self.integrate(np.arange(0, Nsteps * dt, dt))


def main():
    lorenz40 = LorenzModel(dimension=5)
    x0 = 8 * np.ones(lorenz40.dim)
    x0[0] = x0[0] + 0.01
    lorenz40.initial_state(x0)
    t = np.arange(0, 100, 0.01)
    lorenz40.step(0.1, 10)
    plt.contourf(lorenz40.history)
    plt.show()
    plt.plot(lorenz40.history[1, :].T)
    plt.show()


if __name__ == "__main__":
    main()
