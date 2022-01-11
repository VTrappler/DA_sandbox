# -*- coding: utf-8 -*-
#!/usr/bin/env python


from typing import Callable, Tuple, Union, Optional
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import scipy
from dataclasses import dataclass
import tqdm
from DAmethod.EnsembleMethod import EnsembleMethod

from dynamicalsystems.anharmonic_oscillator import NonLinearOscillatorModel

"""
x_{k+1} = M_k(x_k)  + w_k
y_k = H_k(x_k) + v_k

where Cov[v_k] = R_k, Cov[w_k] = Q_k
"""


class EnKF(EnsembleMethod):
    """Wrapper class for running an EnKF"""

    @classmethod
    def Kalman_gain(cls, H: np.ndarray, Pf: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Computes the Kalman Gain Given the observation matrix, the prior covariance matrix and the error covariance matrix error R

        :param H: Linearized observation operator
        :type H: np.ndarray
        :param Pf: Covariance matrix of the prior error
        :type Pf: np.ndarray
        :param R: Covariance matrix of the observation errors
        :type R: np.ndarray
        :return: Kalman Gain
        :rtype: np.ndarray
        """

        return np.linalg.multi_dot(
            [
                Pf,
                H.T,
                np.linalg.inv(np.linalg.multi_dot([H, Pf, H.T]) + R),
            ]
        )

    def __init__(
        self,
        state_dimension: int,
        Nensemble: int,
        R: np.ndarray,
        inflation_factor: float = 1.0,
    ) -> None:
        self._state_dimension = state_dimension
        self._Nensemble = Nensemble
        self._R = R
        self._inflation_factor = inflation_factor

    # EnKF parameters ---

    def analysis(self, y: np.ndarray, stochastic: bool = True) -> None:
        """Performs the analysis step given the observation

        :param y: Observation to be assimilated
        :type y: np.ndarray
        :param stochastic: Perform Stochastic EnKF, ie perturbates observations, defaults to True
        :type stochastic: bool, optional
        """
        if stochastic:
            u = np.random.multivariate_normal(
                mean=np.zeros_like(y), cov=self.R, size=self.Nensemble
            )
            y = np.atleast_2d(y).T + u.T
            self._R = np.atleast_2d(np.cov(u.T))  # Compute empirical covariance matrix
        Kstar = self.Kalman_gain(self.linearH, self.inflation_factor * self.Pf, self.R)
        anomalies_vector = y - self.linearH @ self.xf_ensemble
        self.xa_ensemble = self.xf_ensemble + Kstar @ anomalies_vector
        try:
            self.xa_ensemble_total = np.concatenate(
                [self.xa_ensemble_total, self.xa_ensemble[:, :, np.newaxis]], 2
            )
        except AttributeError:
            self.xa_ensemble_total = self.xa_ensemble[:, :, np.newaxis]

    def forecast_ensemble(self) -> None:
        """Propagates the ensemble members through the model"""
        try:
            self.xf_ensemble = np.apply_along_axis(
                self.forward,
                axis=0,
                arr=self.xa_ensemble,
            )
        except AttributeError:
            self.xf_ensemble = np.apply_along_axis(
                self.forward,
                axis=0,
                arr=self.xf_ensemble,
            )

        self.xf_ensemble_total = np.concatenate(
            [self.xf_ensemble_total, self.xf_ensemble[:, :, np.newaxis]], 2
        )

    def run(
        self,
        Nsteps: int,
        get_obs: Callable[[int], Tuple[float, np.ndarray]],
        full_obs: bool = True,
    ) -> dict:

        observations = []
        ensemble_f = []
        ensemble_a = []
        time = []
        for i in range(Nsteps):
            self.forecast_ensemble()
            ensemble_f.append(self.xf_ensemble)
            t, y = get_obs(i)
            observations.append(y)
            time.append(t)

            self.analysis(self.H(y))
            ensemble_a.append(self.xa_ensemble)
        return {
            "observations": observations,
            "ensemble_f": ensemble_f,
            "ensemble_a": ensemble_a,
            "time": time,
        }
