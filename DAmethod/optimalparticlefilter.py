#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from typing import Optional, Union, Callable, Tuple
import scipy.stats
from DAmethod.baseparticlefilter import BaseParticleFilter
from DAmethod.utils import Kalman_gain


class OptimalKPF(BaseParticleFilter):
    """Implementation of the standard particle filter, where the prior is chosen as sampling density, and the likelihood is used to reweights"""

    def __init__(
        self, state_dimension: int, Nparticles: int, R: np.ndarray, Q: np.ndarray
    ) -> None:
        """initialise the optimal PF (based on EKF)

        :param state_dimension: dimension of the state vector
        :type state_dimension: int
        :param Nparticles: Number of particles to consider
        :type Nparticles: int
        :param R: Observation error covariance matrix
        :type R: np.ndarray
        :param Q: Model error covariance matrix
        :type Q: np.ndarray
        """
        super().__init__(state_dimension, Nparticles, R)
        self.Q = Q

    def sample_proposal(self, obs: np.ndarray):
        """Sample from the proposal distribution

        :param obs: observation to be "assimilated" using Kalman gain
        :type obs: np.ndarray
        """
        self.previous_particles = np.copy(self.particles)
        xk = np.apply_along_axis(
            self.forward,
            axis=0,
            arr=self.particles,
        )
        self.Kstar = Kalman_gain(self.linearH, self.Q, self.R)
        xkbar = (np.eye(self.state_dimension) - self.Kstar @ self.linearH) @ xk + (
            self.Kstar @ obs
        )[:, np.newaxis]
        Pk = (np.eye(self.state_dimension) - self.Kstar @ self.linearH) @ self.Q

        for i, xki in enumerate(xkbar.T):
            self.particles[:, i] = np.random.multivariate_normal(mean=xki, cov=Pk)

    def update_weights(self, y: np.ndarray) -> None:
        """Update the weights using the likelihood (Standard PF/Bootstrap Bayesian Filtering)
        :param y: observation
        :type y: np.ndarray
        """
        lik = np.empty_like(self.weights)
        for i, part in enumerate(self.previous_particles.T):
            cov_lik = self.linearH @ self.Q @ self.linearH.T + self.R
            lik[i] = scipy.stats.multivariate_normal(self.H(part), cov_lik).pdf(y)
        self.weights = self.weights * lik

    def run(
        self,
        Nsteps: int,
        get_obs: Callable[[int], Tuple[float, np.ndarray]],
        full_obs: bool = True,
        ESS_lim: Optional[float] = None,
    ) -> dict:
        """Run the standard PF

        :param Nsteps: Assimilation steps
        :type Nsteps: int
        :param get_obs: callable which gives the observation at i
        :type get_obs: Callable[[int], Tuple[float, np.ndarray]]
        :param full_obs: Does the observations is given in the same space as the state, ie does the observation operator needs to be applied, defaults to True
        :type full_obs: bool, optional
        :param ESS_lim: Given threshold for resampling, defaults to None
        :type ESS_lim: Optional[float], optional
        :return: Dictionary containing all the filtered data
        :rtype: dict
        """
        if ESS_lim is None:
            ESS_lim = 0.6 * self.Nparticles
        observations = []
        estimates = []
        particles = []
        time = []
        weights = []
        print(f"{'Iter': <5}{'ESS': <20}{'RMS': >10}")
        for i in range(Nsteps):
            particles.append(self.particles)
            weights.append(self.weights)

            # Get observations
            t, obs = get_obs(i)
            observations.append(obs)
            time.append(t)
            # Sample from the proposal distribution given by KF
            # Update weights
            if full_obs:
                y = self.H(obs)
                self.sample_proposal(y)
                self.update_weights(y)
            else:
                self.sample_proposal(obs)
                self.update_weights(obs)
            # Normalize weights
            self.normalize_weights()

            estimates.append(self.estimate())
            if self.get_ESS() < ESS_lim:
                self.resample_particles_systematic()
        return {
            "observations": observations,
            "estimates": estimates,
            "particles": particles,
            "time": time,
            "weights": weights,
        }
