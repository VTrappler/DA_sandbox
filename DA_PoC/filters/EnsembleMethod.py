from typing import Callable
import numpy as np
import scipy.linalg as la


class EnsembleMethod:
    @property
    def Nensemble(self):
        return self._Nensemble

    @property
    def state_dimension(self):
        return self._state_dimension

    @property
    def R(self):
        """Observation error covariance matrix"""
        return self._R

    @R.setter
    def R(self, value):
        self._R = value
        self.Rinv = la.inv(value)
        self.Rchol = la.cholesky(value)

    @property
    def H(self):
        """Observation operator or matrix"""
        return self._H

    @H.setter
    def H(self, value):
        if callable(value):
            self._H = value
            self.linearH = None
        elif isinstance(value, np.ndarray):
            self._H = lambda x: value @ x
            self.linearH = value

    # Methods for manipulating ensembles ---
    @property
    def xf_ensemble(self) -> np.ndarray:
        return self._xf_ensemble

    @xf_ensemble.setter
    def xf_ensemble(self, xf_i: np.ndarray):
        self._xf_ensemble = xf_i
        self.Pf = np.cov(xf_i)

    @property
    def xa_ensemble(self) -> np.ndarray:
        return self._xa_ensemble

    @xa_ensemble.setter
    def xa_ensemble(self, xa_i: np.ndarray):
        self._xa_ensemble = xa_i
        self.Pa = np.cov(xa_i)

    def set_forwardmodel(self, model: Callable) -> None:
        self.forward = model

    @property
    def inflation_factor(self):
        """Initialize inflation factor"""
        return self._inflation_factor

    def generate_ensemble(
        self, mean: np.ndarray, cov: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Generation of the ensemble members, using a multivariate normal rv

        :param mean: mean of the ensemble members
        :type mean: np.ndarray
        :param cov: Covariance matrix
        :type cov: np.ndarray
        """
        self.xf_ensemble = rng.multivariate_normal(
            mean=mean, cov=cov, size=self.Nensemble
        ).T
        self.xf_ensemble_total = self.xf_ensemble[:, :, np.newaxis]
