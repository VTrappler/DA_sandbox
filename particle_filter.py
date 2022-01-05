#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import scipy.stats
from lorenz63 import Lorenz63Model

# from lorenz import LorenzModel
from anharmonic_oscillator import NonLinearOscillatorModel
import tqdm


class ParticleFilter:
    def __init__(self, state_dimension, Nparticles, R):
        self._state_dimension = state_dimension
        self._Nparticles = Nparticles
        self._R = R

    @property
    def Nparticles(self):
        return self._Nparticles

    @property
    def state_dimension(self):
        return self._state_dimension

    @property
    def R(self):
        """Observation error covariance matrix"""
        return self._R

    @property
    def H(self):
        """Observation operator or matrix"""
        return self._H

    @H.setter
    def H(self, value):
        if callable(value):
            self._H = value
        elif isinstance(value, np.ndarray):
            self._H = lambda x: value @ x

    @classmethod
    def weighted_moments(cls, X, weights):
        mean = np.average(X, weights=weights, axis=1)
        var = np.average(
            (X - mean.reshape(-1, 1)) ** 2,
            weights=weights,
            axis=1,
        )
        return mean, var

    def generate_particles(self, prior_mean: np.ndarray, prior_cov: np.ndarray):
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
        return 1.0 / ((self.weights ** 2).sum())

    def resample_particles(self):
        new_particles = np.random.choice(
            a=self.Nparticles, size=self.Nparticles, p=self.weights, replace=True
        )
        self.particles = self.particles[:, new_particles]
        self.weights = np.ones(self.Nparticles) / self.Nparticles

    def resample_particles_systematic(self):
        cumulative_weights = np.cumsum(self.weights)
        uj = (
            np.arange(self.Nparticles) / self.Nparticles
            + np.random.uniform() / self.Nparticles
        )
        new_particles = np.digitize(uj, cumulative_weights)
        self.particles = self.particles[:, new_particles]
        self.weights = np.ones(self.Nparticles) / self.Nparticles

    def resample_particles_RSR(self):
        dU0 = np.random.uniform() / self.Nparticles
        new_particles = np.empty(self.Nparticles, dtype="int")
        dUprev = dU0
        for m, w in enumerate(self.weights):
            new_particles[m] = (np.floor((w - dUprev) * self.Nparticles) + 1).astype(
                "int"
            )
            dUprev = dUprev + (new_particles[m] / self.Nparticles) - w
        print(new_particles.min(), new_particles.max())
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
            dist = y - self.H(part)
            lik[i] = scipy.stats.multivariate_normal(np.zeros(len(dist)), self.R).pdf(
                dist
            )
        self.weights = self.weights * lik
        self.weights = self.weights / sum(self.weights)

    def estimate(self):
        return self.weighted_moments(self.particles, self.weights)

    def run(self, Nsteps, get_obs, full_obs=True, ESS_lim=None):
        if ESS_lim is None:
            ESS_lim = 0.6 * self.Nparticles
        observations = []
        estimates = []
        particles = []
        time = []
        weights = []
        print(f"{'Iter': <5}{'ESS': <20}{'RMS': >10}")
        for i in range(Nsteps):
            self.propagate_particles()
            particles.append(self.particles)
            weights.append(self.weights)
            t, obs = get_obs(i)
            observations.append(obs)
            time.append(t)
            if full_obs:
                self.update_weights(self.H(obs))
            else:
                self.update_weights(obs)
            estimates.append(self.estimate())
            RMS = np.sum((self.estimate() - obs) ** 2)
            if self.get_ESS() < ESS_lim:
                print(f"{i: <5}{self.get_ESS():<20.2f}{RMS:>10.2f} Resampling")
                self.resample_particles_systematic()
            else:
                print(f"{i: <5}{self.get_ESS():<20.2f}{RMS:>10.2f}")
        return {
            "observations": observations,
            "estimates": estimates,
            "particles": particles,
            "time": time,
            "weights": weights,
        }


def main_PF_oscillator(
    assimilation_steps=100,
    Nparticles=200,
    period_assim=50,
):
    H = np.atleast_2d([0, 1])
    R = np.atleast_2d([49])
    PF = ParticleFilter(state_dimension=2, Nparticles=Nparticles, R=R)
    PF.H = H
    PF.set_forwardmodel(
        lambda x: NonLinearOscillatorModel.integrate(0, x, period_assim)[1][:, -1]
    )

    truth = NonLinearOscillatorModel()
    truth.set_initial_state(0, [0, 1])

    def generate_observations(i):
        truth.forward(period_assim)
        y = truth.state_vector[:, -1] + np.random.randn(1) * np.sqrt(float(R))
        return truth.t[-1], y

    PF.generate_particles(
        prior_mean=np.array([0, 1]), prior_cov=np.array([[2, 0], [0, 2]])
    )

    dPF = PF.run(assimilation_steps, generate_observations, full_obs=True, ESS_lim=None)

    # observations = []
    # estimates = []
    # particles_ = []
    # t_assim = []
    # weights = []
    # for i in range(assimilation_steps):
    #     PF.propagate_particles()
    #     particles_.append(PF.particles)
    #     weights.append(PF.weights)
    #     tt, obs = generate_observations()
    #     observations.append(obs)
    #     t_assim.append(tt)
    #     PF.update_weights(H @ obs)
    #     # plt.scatter(x=PF.particles[0, :], y=PF.particles[1, :], s=PF.weights)
    #     # plt.plot(obs[0], obs[1], "x", color="red")
    #     est, _ = PF.estimate()
    #     estimates.append(est)
    #     print(f"{i}, ESS={PF.get_ESS()}, sumweights = {PF.weights.sum()}")
    #     if PF.get_ESS() < (10 * PF.Nparticles):
    #         PF.resample_particles()
    #         print("Resampling")
    #     # plt.plot(est[0], est[1], "x", color="black")
    #     # plt.show()

    est_ = np.array(dPF["estimates"])[:, 0]
    obs_ = np.array(dPF["observations"])[:, 0]
    wei_ = np.array(dPF["weights"])

    for i in range(assimilation_steps):
        plt.scatter(
            period_assim * (i + 1) * np.ones(Nparticles),
            np.asarray(dPF["particles"])[i, 1, :],
            marker="o",
            c="cyan",
            s=300 * wei_[i, :],
        )
    v_ = [
        PF.weighted_moments(
            np.array(dPF["particles"])[i, :, :],
            np.array(dPF["weights"])[i, :],
        )[1]
        for i in range(assimilation_steps)
    ]
    std = np.sqrt(np.array(v_)[:, 1])
    plt.fill_between(
        dPF["time"],
        np.squeeze(PF.H(est_.T) + 2 * std),
        np.squeeze(PF.H(est_.T) - 2 * std),
        color="gray",
        alpha=0.3,
    )
    plt.scatter(dPF["time"], obs_, marker="o", c="red", s=20)
    plt.scatter(dPF["time"], PF.H(est_.T), marker="x", color="blue", s=20)
    plt.plot(truth.state_vector[1, :])
    plt.show()


def main_Lorenz63_oscillator(
    assimilation_steps=100,
    Nparticles=200,
    period_assim=50,
):
    dt = 0.02
    Lorenz63Model.dt = dt
    H = np.atleast_2d([[1, 0, 0], [0, 1, 0]])
    sigobs = 5
    R = sigobs * np.atleast_2d([[1, 0], [0, 1]])
    PF = ParticleFilter(state_dimension=3, Nparticles=Nparticles, R=R)
    PF.H = H
    PF.set_forwardmodel(lambda x: Lorenz63Model.integrate(0, x, period_assim)[1][:, -1])

    truth = Lorenz63Model()
    truth.set_initial_state(-5000 * dt, np.array([0, 1, 0]))
    truth.forward(5000)  # Burn-in period

    def generate_observations(i):
        truth.forward(period_assim)
        y = truth.state_vector[:, -1] + np.random.randn(1) * sigobs
        return truth.t[-1], y

    PF.generate_particles(
        prior_mean=truth.state_vector[:, -1] + np.random.randn(3) * sigobs,
        prior_cov=sigobs / 3.0 * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )
    # for i, vec in enumerate(truth.state_vector):
    #     plt.subplot(3, 1, i + 1)
    #     plt.plot(vec)
    # plt.show()

    dPF = PF.run(assimilation_steps, generate_observations, full_obs=True, ESS_lim=None)

    est_ = np.array(dPF["estimates"])[:, 0, :]
    obs_ = np.array(dPF["observations"])[:, 0]
    wei_ = np.array(dPF["weights"])

    for i in range(assimilation_steps):
        plt.scatter(
            period_assim * (i + 1) * np.ones(Nparticles) * dt,
            np.asarray(dPF["particles"])[i, 0, :],
            marker="o",
            c="cyan",
            s=300 * wei_[i, :],
        )
    v_ = [
        PF.weighted_moments(
            np.array(dPF["particles"])[i, :, :],
            np.array(dPF["weights"])[i, :],
        )[1]
        for i in range(assimilation_steps)
    ]
    # std = np.sqrt(np.array(v_))[:, 0]
    # plt.fill_between(
    #     dPF["time"],
    #     np.squeeze(PF.H(est_.T) + 2 * std),
    #     np.squeeze(PF.H(est_.T) - 2 * std),
    #     color="gray",
    #     alpha=0.3,
    # )
    # plt.scatter(dPF["time"], obs_, marker="o", c="red", s=20)
    # plt.scatter(dPF["time"], PF.H(est_.T), marker="x", color="blue", s=20)
    # plt.plot(truth.t[5001:], truth.state_vector[0, 5001:])
    # plt.plot(truth.t[5001:], truth.state_vector[1, 5001:])

    # plt.show()
    return dPF, truth


if __name__ == "__main__":
    main_PF_oscillator(Nparticles=1000, assimilation_steps=150, period_assim=50)
    dPF, truth = main_Lorenz63_oscillator(
        Nparticles=1000, assimilation_steps=50, period_assim=5
    )

    est_ = np.array(dPF["estimates"])[:, 0, :]
    obs_ = np.array(dPF["observations"])

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(truth.t[5001:], truth.state_vector[i, 5001:])
        plt.scatter(dPF["time"], obs_[:, i], marker="o", c="red", s=20)
        plt.scatter(dPF["time"], est_[:, i], marker="x", color="blue", s=20)
        plt.vlines(dPF["time"], est_[:, i], obs_[:, i])
    plt.show()
