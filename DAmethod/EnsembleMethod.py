from typing import Callable
import numpy as np


class EnsembleMethod:
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
