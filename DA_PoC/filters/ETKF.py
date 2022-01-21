from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import scipy.linalg as la
from .EnsembleMethod import EnsembleMethod
from numpy.linalg import multi_dot
from tqdm.autonotebook import tqdm

"""
x_{k+1} = M_k(x_k)  + w_k
y_k = H_k(x_k) + v_k

where Cov[v_k] = R_k, Cov[w_k] = Q_k
"""


class ETKF(EnsembleMethod):
    """Wrapper class for running an Ensemble Transform Kalman Filter"""

    def __init__(
        self,
        state_dimension: int,
        Nensemble: int,
        R: np.ndarray,
        inflation_factor: float = 1.0,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> None:
        self._state_dimension = state_dimension
        self._Nensemble = Nensemble
        self._R = R
        self._inflation_factor = inflation_factor

    def state_anomalies(self) -> np.ndarray:
        """Computes the state normalised anomalies x - xbar/ sqrt(N-1)

        :return: matrix of state anomalies X of dim (state_dim * Nensemble)
        :rtype: np.ndarray
        """
        return (self.xf_ensemble - self.xf_ensemble.mean(1, keepdims=True)) / np.sqrt(
            self.Nensemble - 1
        )

    def observation_anomalies(self) -> np.ndarray:
        """Computes the observation normalised anomalies H(x) - Hbar(x) / sqrt(N-1)

        :return: matrix of state anomalies Y of dim (obs_dim * Nensemble)
        :rtype: np.ndarray
        """
        Hxf = self.H(self.xf_ensemble)
        Yf = (Hxf - Hxf.mean(1, keepdims=True)) / np.sqrt(self.Nensemble - 1)
        precYf = la.solve_triangular(self.Rchol, Yf)
        return Yf, precYf

    def compute_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the transform matrix using "naive" method

        :return: transform matrix T and its square
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        Yf, Yfhat = self.observation_anomalies()
        Tsqm1 = np.eye(self.Nensemble) + Yf.T @ self.Rinv @ Yf
        print(f"{la.det(Tsqm1)=}")
        Tsq = la.inv(Tsqm1)
        return la.sqrtm(Tsq), Tsq

    def compute_transform_SVD(self) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the transform matrix using SVD

        :return: Transform matrix, and the SVD of the preconditionned Yf
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        Yf, Yfhat = self.observation_anomalies()
        U, Sigma, VT = la.svd(Yfhat.T)
        if self.linearH.shape[0] < self.Nensemble:
            Lm = np.concatenate(
                [
                    1 / np.sqrt(1 + Sigma ** 2),
                    np.ones(self.Nensemble - self.linearH.shape[0]),
                ]
            )
            # Sigma = np.concatenate(
            #     [Sigma, np.zeros(self.Nensemble - self.linearH.shape[0])]
            # )
        else:
            Lm = 1 / np.sqrt(1 + Sigma ** 2)
        T = (U * Lm) @ U.T
        return T, U, Sigma, VT

    def analysis(self, obs: np.ndarray) -> None:
        """Performs the analysis step given the observation

        :param obs: Observation to be assimilated
        :type obs: np.ndarray
        """
        innovation_vector = obs - (self.H(self.xf_ensemble)).mean(1)
        Yf, Yfhat = self.observation_anomalies()
        transform_matrix, U, Sigma, VT = self.compute_transform_SVD()
        omega = transform_matrix @ transform_matrix.T
        # wa = T2 @ Yf.T @ la.solve(self.R, innovation_vector, sym_pos=True)
        # wa = omega @ Yf.T @ self.Rinv @ innovation_vector
        xfbar = self.xf_ensemble.mean(1, keepdims=True)
        Xa = np.sqrt(self.Nensemble - 1) * self.state_anomalies() @ transform_matrix
        # print(f"{transform_matrix.shape=}, {wa.shape=}")

        wa = multi_dot(
            [
                U,
                la.diagsvd(Sigma, self.Nensemble, self.linearH.shape[0]),
                np.diag(1 / (1 + Sigma ** 2)),
                VT,
                la.solve_triangular(self.Rchol, innovation_vector),
            ]
        )

        # print(f"{wa.shape=}")
        # print(f"{xfbar.shape=}")
        # print(f"{self.state_anomalies().shape=}")
        # print(f"{Xa.shape=}")
        xabar = xfbar + self.state_anomalies() @ wa[:, np.newaxis]
        # self.xa_ensemble = xfbar + self.inflation_factor * self.state_anomalies() @ (
        #     wa + np.sqrt(self.Nensemble - 1) * transform_matrix
        # )
        self.xa_ensemble = xabar + Xa
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
        observations = []
        ensemble_f = []
        ensemble_a = []
        time = []
        if not verbose:
            iterator = range(Nsteps)
        else:
            iterator = tqdm(range(Nsteps))
        for i in iterator:
            self.forecast_ensemble()
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
