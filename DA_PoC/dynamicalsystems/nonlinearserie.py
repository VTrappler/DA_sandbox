#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from solvers.solvers import integrate_step
from dynamicalsystems.dynmodels import DynamicalModel as Model


class NonLinearSerie(Model):
    def __init__(self):
        pass

    @classmethod
    def step(cls, f, t, x, dt):
        return np.array(0.5 * x + 25 * (x / (1 + x ** 2)) + 8 * np.cos(1.2 * t))

    @classmethod
    def integrate(cls, t0, x0, Nsteps):
        return integrate_step(cls.step, f=None, t0=t0, x0=x0, dt=cls.dt, Nsteps=Nsteps)

    @classmethod    
    def observation(cls, x):
        return np.array(x**2 / 20)
serie = NonLinearSerie()
serie.set_initial_state(0, np.array([0, 1]))
serie.forward(1000)
plt.plot(serie.state_vector[0, :])
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
    # main()
    pass
