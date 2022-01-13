from typing import Callable, Tuple, Union
import numpy as np
import scipy.linalg as la
from utils import Kalman_gain
from tqdm.autonotebook import tqdm


class KalmanFilter:
    def __init__(self, state_dimension: int) -> None:
        self.xf = np.empty((state_dimension, 0))
        self.xa = np.empty((state_dimension, 0))
        self.state_dimension = state_dimension

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
    def Q(self):
        """Model error covariance matrix"""
        return self._Q

    @Q.setter
    def Q(self, value):
        self._Q = value
        self.Qinv = la.inv(value)
        self.Qchol = la.cholesky(value)

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

    @property
    def M(self):
        """Model operator or matrix"""
        return self._M

    @M.setter
    def M(self, value):
        if callable(value):
            self._M = value
            self.linearM = None
        elif isinstance(value, np.ndarray):
            self._M = lambda x: value @ x
            self.linearM = value

    def set_x0(self, x0, P0):
        self.xf = x0
        self.xa = x0
        self.Pf = P0
        self.Pa = P0

    def analysis(self, obs: Union[np.ndarray, None]) -> None:
        if obs is None:
            self.xa = self.xf
            self.Pa = self.Pf
        else:
            innovation_vector = obs - self.H(self.xf)
            Kstar = Kalman_gain(self.linearH, self.Pf, self.R)
            self.xa = self.xf + Kstar @ innovation_vector
            self.Pa = (np.eye(self.state_dimension) - Kstar @ self.linearH) @ self.Pf

    def forecast(self) -> None:
        try:
            self.xf = self.M(self.xa)
            self.Pf = self.linearM @ self.Pa @ self.linearM.T + self.Q
        except:
            self.xf = self.M(self.xf)
            self.Pf = self.linearM @ self.Pf @ self.linearM.T + self.Q

    def run(
        self,
        Nsteps: int,
        get_obs: Callable[[int], Tuple[float, np.ndarray]],
    ) -> dict:
        observations = []
        xf = []
        xa = []
        time = []
        for i in tqdm(range(Nsteps)):
            self.forecast()
            print(self.xf)
            xf.append(self.xf)
            t, y = get_obs(i)
            observations.append(y)
            time.append(t)
            self.analysis(self.H(y))
            xa.append(self.xa)
        return {
            "observations": observations,
            "xf": xf,
            "xa": xa,
            "time": time,
        }
