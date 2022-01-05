from abc import ABC, abstractmethod
import numpy as np


class DynamicalModel(ABC):
    """
    A DAModel is defined as
    x_{t+1} = M_t(x_t) + w_t
    y_t = H_t(x_t) + v_k

    Cov[v_t] = R_t -> Observation Error
    Cov[w_t] = Q_k -> Model Error
    """

    def set_initial_state(self, t0, x0, force_reset=False):
        if hasattr(self, "state_vector") and not force_reset:
            raise Exception(
                "This model has already been forwarded. Set 'force_reset' to True in order to overwrite"
            )
        else:
            self.state_vector = x0.reshape(self.dim, 1)
            self.t = np.array(t0).reshape(1)

    def forward(self, Nsteps):
        t0 = self.t[-1]
        x0 = self.state_vector[:, -1]
        t_, x_ = self.integrate(t0, x0, Nsteps)
        self.t = np.concatenate([self.t, t_[1:]])
        self.state_vector = np.concatenate([self.state_vector, x_[:, 1:]], axis=1)

    @abstractmethod
    def integrate(cls, t0, x0, Nsteps):
        pass
