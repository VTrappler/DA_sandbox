# -*- coding: utf-8 -*-
#!/usr/bin/env python


import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.stats
from dataclasses import dataclass
from lorenz import LorenzModel
from anharmonic_oscillator import NonLinearOscillatorModel_2d
import tqdm


class ParticleFilter:
    def __init__(self):
        pass

    @property
    def Nparticles(self):
        return self._Nparticles

    @property
    def state_dimension(self):
        return self._state_dimension

    @property
    def R(self):
        return self._R

    def generate_particles(self, prior_mean, prior_cov):
        self.particles = np.random.multivariate_normal(
            mean=prior_mean,
            cov=prior_cov,
            size=self.Nparticles,
        ).T
        self.weights = np.ones(self.Nparticles) / self.Nparticles

    def set_forwardmodel(self, model):
        self.forward = model

    def get_ESS(self):
        """Compute the Effective Sample Size of the particles"""
        return self.Nparticles / (self.weights ** 2).sum()

    def resample_particles(self):
        new_particles = np.random.choice(
            a=self.Nparticles, size=self.Nparticles, p=self.weights
        )
        self.particles = self.particles[:, new_particles]
        self.weights = np.ones(self.Nparticles) / self.Nparticles

    def propagate_particles(self):
        self.particles = np.apply_along_axis(
            self.forward,
            axis=0,
            arr=self.particles,
        )

    def update_weights(self, y):
        lik = np.ones(self.Nparticles)
        for i, part in enumerate(self.particles.T):
            dist = (y - H @ part) ** 2
            lik[i] = scipy.stats.norm(0, self.R).pdf(dist)
        self.weights = self.weights * lik
        self.weights = self.weights / sum(self.weights)

    def estimate(self):
        mean = np.average(self.particles, weights=self.weights, axis=1)
        var = np.average(
            (self.particles - mean.reshape(2, 1)) ** 2,
            weights=self.weights,
            axis=1,
        )
        return mean, var


def main_PF_oscillator(
    assimilation_steps=100,
    Nparticles=200,
    period_assim=50,
):

    H = np.atleast_2d([0, 1])

    def forward(x):
        oscil = NonLinearOscillatorModel_2d()
        oscil.initial_state(x)
        oscil.step(50)
        return oscil.state_vector[:, -1]

    PF = ParticleFilter()
    PF._Nparticles = Nparticles
    PF.set_forwardmodel(forward)
    PF._R = np.atleast_2d([256])

    oscillator = NonLinearOscillatorModel_2d()
    oscillator.initial_state([0, 1])

    def generate_observations():
        oscillator.step(50)
        y = oscillator.state_vector[:, -1] + np.random.randn(1) * np.sqrt(256)
        return y

    PF.generate_particles(
        prior_mean=np.squeeze(H), prior_cov=np.array([[5, 2], [2, 5]])
    )

    observations = []
    estimates = []
    particles_ = []
    for i in range(20):
        PF.propagate_particles()
        particles_.append(PF.particles)
        obs = generate_observations()
        observations.append(obs)
        PF.update_weights(H @ obs)
        print(PF.get_ESS())
        # plt.scatter(x=PF.particles[0, :], y=PF.particles[1, :], s=PF.weights)
        # plt.plot(obs[0], obs[1], "x", color="red")
        est, _ = PF.estimate()
        estimates.append(est)
        if PF.get_ESS() < 1000:
            PF.resample_particles()
        # plt.plot(est[0], est[1], "x", color="black")

        # plt.show()


est_ = np.array(estimates)[:, 0]
obs_ = np.array(observations)[:, 0]
for i in range(20):
    plt.plot(i * np.ones(200), np.asarray(particles_)[i, 0, :], ".")
plt.plot(est_, "x", color="blue")
plt.plot(obs_, "o", color="red")
plt.show()
if __name__ == "__main__":
    main_PF_oscillator()
