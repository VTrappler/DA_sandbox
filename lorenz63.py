#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import scipy


class Lorenz63Model:
    def __init__(self):
        self.dim = 3
        self.state = np.empty((self.dim, 1))
        # self.history = np.empty(0), np.empty((self.dim, 0))

    def initial_state(self, init):
        self.initstate = init
        self.history = np.atleast_1d(0), init.reshape(3, 1)

    @staticmethod
    def L63(x, t):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        d = np.zeros(3)
        d[0] = 10 * (x[1] - x[0])
        d[1] = 28 * x[0] - x[1] - x[0] * x[2]
        d[2] = -x[2] * 8.0 / 3.0 + x[1] * x[0]
        return d

    def Jacobian(self, x):
        return np.array(
            [
                [-10, 10, 0],
                [28 - x[2], -1 - x[0]],
                [x[1], x[0], -8.0 / 3.0],
            ]
        )

    def integrate(self, t):
        integration = scipy.integrate.odeint(
            func=self.L63, y0=self.history[1][:, -1], t=t
        )
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
    lorenz63 = Lorenz63Model()
    dt = 0.02
    lorenz63.initial_state(np.array([0, 1, 0]))
    lorenz63.step(dt, 2000)

    plt.plot(lorenz63.history[1].T)
    plt.show()
    H = np.array([1, 0, 0])

    def generate_obs(Ntot, burn=1000):
        lorenz63 = Lorenz63Model()
        dt = 0.02
        lorenz63.initial_state(np.array([0, 1, 0]))
        lorenz63.step(dt, Ntot)
        xobs = H @ lorenz63.history[1]
        xobs_per = xobs + np.random.normal(loc=0, scale=np.sqrt(0.2), size=(Ntot))
        return xobs[burn:], xobs_per[burn:]
