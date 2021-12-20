# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class NonlinearOscillatorModel:
    def __init__(
        self,
    ):
        "docstring"

    def initial_state(self, x0=0, x1=1):
        self.state_vector = np.atleast_2d([x0, x1])

    def nonlinear_oscillator_step(self, xk, xkm1, xi=None, omega=0.035, lam=3e-5):
        if xi is None:
            xi = np.random.randn(1) * np.sqrt(0.0025)
        return 2 * xk - xkm1 + (omega ** 2) * xk - (lam ** 2) * xk ** 3 + xi

    def step(self, Nsteps):
        xnew = np.empty(Nsteps)
        xnew[0] = self.nonlinear_oscillator_step(
            self.state_vector[-1], self.state_vector[-2]
        )
        xnew[1] = self.nonlinear_oscillator_step(xnew[0], self.state_vector[1])
        for i in range(1, Nsteps - 1):
            xnew[i + 1] = self.nonlinear_oscillator_step(xnew[i], xnew[i - 1])

        self.state_vector = np.concatenate([self.state_vector, xnew])


class NonLinearOscillatorModel_2d:
    def __init__(self):
        "docstring"

    def initial_state(self, initial_state):
        self.state_vector = np.array(initial_state).reshape(2, 1)

    def nonlinear_oscillator_step(self, state, xi=None, omega=0.035, lam=3e-5):
        if xi is None:
            xi = np.random.randn(1) * np.sqrt(0.0025)
        return np.array(
            [
                state[1],
                2 * state[1]
                - state[0]
                + (omega ** 2) * state[1]
                - (lam ** 2) * state[1] ** 3
                + xi[0],
            ]
        )

    def step(self, Nsteps):
        xnew = np.empty((2, Nsteps))
        # xnew[0] = self.nonlinear_oscillator_step(self.state_vector)
        xnew[:, 0] = self.nonlinear_oscillator_step(self.state_vector[:, -1])

        for i in range(1, Nsteps):
            xnew[:, i] = self.nonlinear_oscillator_step(xnew[:, i - 1])
        self.state_vector = np.hstack([self.state_vector, xnew])


osci_2d = NonLinearOscillatorModel_2d()
osci_2d.initial_state([0, 1])
osci_2d.step(10)
osci_2d.state_vector


def sample_observations(vector, spacing, shift):
    return vector[shift::spacing]


def main():
    N_iter = 5000
    oscillator = NonLinearOscillatorModel_2d()
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
