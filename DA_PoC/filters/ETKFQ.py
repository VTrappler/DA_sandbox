# -*- coding: utf-8 -*-


from typing import Callable, Tuple
import numpy as np
import scipy.linalg as la
from dataclasses import dataclass

from tqdm.autonotebook import tqdm

from .ETKF import ETKF


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
        inflation_factor: float = 1,
    ) -> None:
        if state_dimension < Nensemble:
            raise ValueError(
                "State dimension should be larger than the ensemble number"
            )
        super().__init__(state_dimension, Nensemble, R, inflation_factor)
        self.Q = Q
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

    def update_ensemble(self) -> np.ndarray:
        """Update ensemble using the deviation matrix taking into account the model error

        :return: updated ensemble
        :rtype: np.ndarray
        """

        print(f"{self.U.shape=}")
        m = self.Nensemble
        print(f"{m=}")
        tmp = self.xf_ensemble @ self.U
        xkbar, deviation_mat = tmp[:, 0], tmp[:, 1:]
        eivals, Vk = np.linalg.eigh(deviation_mat @ deviation_mat.T + self.Q)
        # print(f"{eivals.shape=}")
        # print(f"{Vk.shape=}")
        # print(f"{eivals=}")
        truncated_eigvals = eivals[:-(m):-1]
        print(f"{truncated_eigvals=}")
        Lambdak = np.diag(truncated_eigvals)  # Keep m-1 largest eigenvalues
        # print(f"{Lambdak.shape=}")
        Vk = Vk[:, :-(m):-1]
        # print(f"{Vk.shape=}")
        new_deviation_matrix = Vk @ np.sqrt(Lambdak)
        # print(f"{xkbar.shape=}")
        # print(f"{deviation_mat.shape=}")
        # print(f"{new_deviation_matrix.shape=}")
        # print(f"{xkbar[:, np.newaxis].shape=}")

        Ek = (
            np.concatenate([xkbar[:, np.newaxis], new_deviation_matrix], axis=1)
            @ self.Uinv
        )
        return Ek

    def run(
        self,
        Nsteps: int,
        get_obs: Callable[[int], Tuple[float, np.ndarray]],
        full_obs: bool = True,
        verbose: bool = True,
    ) -> dict:
        """Run the filter

        :param Nsteps: Number of assimilation steps to perform
        :type Nsteps: int
        :param get_obs: Function which provides the (time, observation) tuple
        :type get_obs: Callable[[int], Tuple[float, np.ndarray]]
        :param full_obs: Does the obs operator needs to be applied before the analysis, defaults to True
        :type full_obs: bool, optional
        :param verbose: tqdm progress bar, defaults to True
        :type verbose: bool, optional
        :return: Dictionary containing the ensemble members, analised or not
        :rtype: dict
        """
        if verbose:
            iterator = tqdm(range(Nsteps))
        else:
            iterator = range(Nsteps)

        observations = []
        ensemble_f = []
        ensemble_a = []
        time = []
        for i in iterator:
            self.forecast_ensemble()
            self.xf_ensemble = self.update_ensemble()
            ensemble_f.append(self.xf_ensemble)
            t, y = get_obs(i)
            observations.append(y)
            time.append(t)

            if full_obs:
                self.analysis(self.H(y))
            else:
                self.analysis(y)

            ensemble_a.append(self.xa_ensemble)
        return {
            "observations": observations,
            "ensemble_f": ensemble_f,
            "ensemble_a": ensemble_a,
            "time": time,
        }
