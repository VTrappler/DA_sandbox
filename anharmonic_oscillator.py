#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from solvers import integrate_step
from dynmodels import DynamicalModel as Model


class NonLinearOscillatorModel(Model):
    dim = 2
    omega = 0.035
    lam = 3e-5
    xi = None
    dt = 1

    def __init__(self):
        pass

    def set_initial_state(self, t0, x0):
        self.state_vector = np.array(x0).reshape(2, 1)
        self.t = np.array(t0).reshape(1)

    @classmethod
    def step(cls, f, t, x, dt):
        if cls.xi is None:
            xi = np.random.randn(1) * np.sqrt(0.0025)
        else:
            xi = cls.xi
        return np.array(
            [
                x[1],
                2 * x[1]
                - x[0]
                + (cls.omega ** 2) * x[1]
                - (cls.lam ** 2) * x[1] ** 3
                + xi[0],
            ]
        )

    @classmethod
    def integrate(cls, t0, x0, Nsteps):
        return integrate_step(cls.step, f=None, t0=t0, x0=x0, dt=cls.dt, Nsteps=Nsteps)


osci = NonLinearOscillatorModel()
osci.set_initial_state(0, [0, 1])
osci.forward(1000)
plt.plot(osci.state_vector[0, :])
plt.show()


def sample_observations(vector, spacing, shift):
    return vector[shift::spacing]


def main():
    N_iter = 5000
    oscillator = NonLinearOscillatorModel()
    oscillator.initial_state([0, 1])
    oscillator.step(N_iter)
    sigma2_obs = 49
    observations = oscillator.state_vector[:, 0] + np.random.randn(
        len(oscillator.state_vector)
    ) * np.sqrt(sigma2_obs)
    plt.plot(observations)
    plt.show()


if __name__ == "__main__":
    main()
