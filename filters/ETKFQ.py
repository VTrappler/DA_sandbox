# -*- coding: utf-8 -*-


from typing import Callable, Tuple, Union, Optional
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from dataclasses import dataclass

from scipy.sparse import construct
from DAmethod.EnsembleMethod import EnsembleMethod
from DAmethod.ETKF import ETKF


"""
x_{k+1} = M_k(x_k)  + w_k
y_k = H_k(x_k) + v_k

where Cov[v_k] = R_k, Cov[w_k] = Q_k
"""


class ETKFQ(ETKF):
    """Wrapper class for running an ETKF-Q, meaning that we assume error in the model, with covariance matrix Q"""

    def __init__(
        self,
        state_dimension: int,
        Nensemble: int,
        R: np.ndarray,
        Q: np.ndarray,
        inflation_factor: float = 1.0,
    ) -> None:
        if state_dimension < Nensemble:
            raise ValueError(
                "State dimension should be larger than the ensemble number"
            )
        self._state_dimension = state_dimension
        self._Nensemble = Nensemble
        self._R = R
        self._inflation_factor = inflation_factor
        self.U, self.Uinv = self.constructU(self.Nensemble)

    @classmethod
    def constructU(cls, m: int) -> Tuple[np.ndarray, np.ndarray]:
        Um = np.zeros((m, m - 1))
        for i in range(m - 1):
            Um[: (i + 1), i] = 1
            Um[i + 1, i] = -(i + 1)
            Um[(i + 2) :, i] = 0
        Um = Um / np.sqrt((Um ** 2).sum(0, keepdims=True))
        U = np.concatenate([np.ones((m, 1)) / m, Um / np.sqrt(m - 1)], axis=1)
        Uinv = np.concatenate([np.ones((m, 1)), Um * np.sqrt(m - 1)], axis=1)
        return U, Uinv

    def normalised_anomalies(self) -> np.ndarray:
        """Computes the state normalised anomalies

        :return: (state_dim x Nensemble) matrix
        :rtype: np.ndarray
        """
        return (self.xf_ensemble - self.xf_ensemble.mean(1, keepdims=True)) / np.sqrt(
            self.Nensemble - 1
        )

    def observation_anomalies(self) -> np.ndarray:
        """Computes the observation anomalies

        :return: the normalised observation anomalies of dimension (obs_dim x Nensemble)
        :rtype: np.ndarray
        """
        Hxf = self.H(self.xf_ensemble)
        return (Hxf - Hxf.mean(1, keepdims=True)) / np.sqrt(self.Nensemble - 1)

    def compute_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the transform matrix T, and its square T2 based on the observation anomalies

        :return: T and (T @ T.T)
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        Yf = self.observation_anomalies()
        Tsqm1 = np.eye(self.Nensemble) + Yf.T @ la.inv(self.R) @ Yf
        Tsq = la.inv(Tsqm1)
        return la.sqrtm(Tsq), Tsq

    def update_ensemble(self) -> np.ndarray:
        """Update ensemble using the deviation matrix taking into account the model error

        :return: the updated ensemble
        :rtype: np.ndarray
        """
        m = self.Nensemble
        tmp = self.xf_ensemble @ self.U
        xkbar, deviation_mat = tmp[:, 0], tmp[:, 1:]
        eivals, Vk = np.linalg.eigh(deviation_mat @ deviation_mat.T + self.Q)
        truncated_eigvals = eivals[: (m - 1) : -1]
        Lambdak = np.diag(truncated_eigvals)  # Keep m-1 largest eigenvalues
        Vk = Vk[:, : (m - 1) : -1]
        new_deviation_matrix = Vk @ np.sqrt(Lambdak)
        Ek = np.concatenate([xkbar, new_deviation_matrix], axis=1) @ self.Uinv
        return Ek

    def analysis(self, y: np.ndarray, stochastic: bool = True) -> None:
        """Performs the analysis step given the observation

        :param y: Observation to be assimilated
        :type y: np.ndarray
        :param stochastic: Perform Stochastic EnKF, ie perturbates observations, defaults to True
        :type stochastic: bool, optional
        """

        innovation_vector = y - (self.H(self.xf_ensemble)).mean(1, keepdims=True)
        Yf = self.observation_anomalies()
        T, T2 = self.compute_transform()
        wa = T2 @ Yf.T @ la.solve(self.R, innovation_vector, sym_pos=True)

        xfbar = self.xf_ensemble.mean(1, keepdims=True)

        self.xa_ensemble = (
            xfbar
            + self.inflation_factor
            * self.normalised_anomalies()
            @ (wa + np.sqrt(self.Nensemble - 1) * T)
        )

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

        self.xf_ensemble = self.update_ensemble()
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
