# -*- coding: utf-8 -*-
#!/usr/bin/env python


import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import scipy
from dataclasses import dataclass
from lorenz import LorenzModel
from anharmonic_oscillator import NonLinearOscillatorModel_2d
import tqdm

"""
x_{k+1} = M_k(x_k)  + w_k
y_k = H_k(x_k) + v_k

where Cov[v_k] = R_k, Cov[w_k] = Q_k
"""


@dataclass
class EnKF:
    state_dimension: int
    ensemble_size: int
    period_assim: int
    H: np.ndarray
    R: np.ndarray

    def __post_init__(self):
        self.observations = []

    @property
    def xf_ensemble(self):
        return self._xf_ensemble

    @xf_ensemble.setter
    def xf_ensemble(self, xf_i):
        self._xf_ensemble = xf_i
        self.Pf = np.cov(xf_i)

    @property
    def xa_ensemble(self):
        return self._xa_ensemble

    @xa_ensemble.setter
    def xa_ensemble(self, xa_i):
        self._xa_ensemble = xa_i
        self.Pa = np.cov(xa_i)

    def set_forwardmodel(self, model):
        self.forwardmodel = model

    def generate_ensemble(self, mean, cov):
        self.xf_ensemble = np.random.multivariate_normal(
            mean=mean, cov=cov, size=self.ensemble_size
        ).T
        self.xf_ensemble_total = self.xf_ensemble[:, :, np.newaxis]

    def analysis(self, y, stochastic=True):
        if stochastic:
            u = np.random.multivariate_normal(
                mean=np.zeros_like(y), cov=self.R, size=self.ensemble_size
            )
            y = y + u
            self.R = np.atleast_2d(np.cov(u.T))
        Kstar = Kalman_gain(self.H, self.Pf, self.R)
        anomalies_vector = y.T - self.H @ self.xf_ensemble
        self.xa_ensemble = self.xf_ensemble + Kstar @ anomalies_vector
        try:
            self.xa_ensemble_total = np.concatenate(
                [self.xa_ensemble_total, self.xa_ensemble[:, :, np.newaxis]], 2
            )
        except AttributeError:
            self.xa_ensemble_total = self.xa_ensemble[:, :, np.newaxis]

    def forecast(self, xin):
        if xin is None:
            xin = self.xf_ensemble
        x_forward = np.empty_like(xin)
        for i, x_member in enumerate(xin.T):
            x_forward[:, i] = self.forwardmodel(x_member)
        self.xf_ensemble = x_forward
        self.xf_ensemble_total = np.concatenate(
            [self.xf_ensemble_total, self.xf_ensemble[:, :, np.newaxis]], 2
        )

    def get_observation(self):
        pass

    def assimilation(self, Nsteps):
        for i in tqdm.trange(Nsteps):
            for j in tqdm.trange(self.period_assim):
                if j == 0 and i != 0:
                    self.forecast(self.xa_ensemble)
                else:
                    self.forecast(self.xf_ensemble)
            y = self.get_observation()
            try:
                self.observations.append(y)
            except AttributeError:
                self.observations = [y]
            self.analysis(y)


def Kalman_gain(H, Pf, R):
    return np.linalg.multi_dot(
        [
            Pf,
            H.T,
            np.linalg.inv(np.linalg.multi_dot([H, Pf, H.T]) + R),
        ]
    )


# Create member of the ensemble


def forecast_lorenz(x_ensemble, ClassModel):
    dim = ClassModel.dim
    N = x_ensemble.shape[0]
    xf_i = np.empty((N, dim))
    for i, x in enumerate(x_ensemble):
        lorenz40_ensemble = LorenzModel(dim)
        lorenz40_ensemble.initial_state(x)
        lorenz40_ensemble.step(0.1, 2)
        xf_i[i, :] = lorenz40_ensemble.history[1][:, -1]
    xf_bar = np.mean(xf_i, 0)
    Pf = np.cov(xf_i.T)
    return xf_i, xf_bar, Pf


def forecast_oscillator(x_ensemble, steps=1):
    N = x_ensemble.shape[1]
    xf_i = np.empty((2, N))
    for i, x in enumerate(x_ensemble.T):
        osci_ensemble = NonLinearOscillatorModel_2d()
        osci_ensemble.initial_state(x)
        osci_ensemble.step(steps)
        xf_i[:, i] = osci_ensemble.state_vector[:, -1]
    xf_bar = np.mean(xf_i, 1)
    Pf = np.cov(xf_i)
    return xf_i, xf_bar, Pf


def analysis(xf_i, y, H, Pf, R, N, stoch=False):
    if stoch:
        u = np.random.multivariate_normal(mean=np.zeros_like(y), cov=R, size=N)
        y = y + u
        R = np.cov(u.T)
    Kstar = Kalman_gain(H, Pf, R)
    anomalies_vector = y.T - H @ xf_i
    xa_i = xf_i + (Kstar @ anomalies_vector)
    # except ValueError:
    #     anomalies_vector = np.squeeze(y) - H @ xf_i.T
    #     xa_i = np.squeeze(xf_i + (Kstar * anomalies_vector).T)
    xa_bar = np.mean(xa_i, 1)
    Pa = np.cov(xa_i.T)
    return xa_i, xa_bar, Pa


def EnKF_lorenz(steps, N, dim, stoch):
    xf = np.empty((N, dim, steps))
    xa = np.empty((N, dim, steps))
    obs = np.empty((dim, steps))
    lorenz40 = LorenzModel(dim)
    x0 = 8 * np.ones(lorenz40.dim)
    x0[0] = x0[0] + 0.01
    lorenz40.initial_state(x0)
    lorenz40.step(0.1, 2)
    y = lorenz40.history[1][:, -1]
    iter_step = 0
    x_ensemble = scipy.random.multivariate_normal(
        mean=8 * np.ones(dim), cov=0.5 * np.diag(np.ones(dim)), size=N
    )
    R = 0.5 * np.diag(np.ones(dim))  # Observational covariance error matrix
    H = np.diag(np.ones(dim))

    obs[:, iter_step] = y
    xf_i, xf_bar, Pf = forecast_lorenz(x_ensemble, LorenzModel(dim))
    xa_i, xa_bar, Pa = analysis(xf_i, y, H, Pf, R, N, stoch)

    xf[:, :, iter_step] = xf_i
    xa[:, :, iter_step] = xa_i

    iter_step = 1
    while iter_step < steps:
        xf_i, xf_bar, Pf = forecast_lorenz(xa_i, LorenzModel(dim))
        lorenz40.step(0.1, 2)
        y = lorenz40.history[1][:, -1]
        xa_i, xa_bar, Pa = analysis(xf_i, y, H, Pf, R, N, stoch)
        xf[:, :, iter_step] = xf_i
        xa[:, :, iter_step] = xa_i
        obs[:, iter_step] = y
        iter_step += 1
    return xf, xa, obs


def main_lorenz40(steps=10, N=200, dim=40, stoch=True):
    xf, xa, y = EnKF_lorenz(steps=steps, N=N, dim=dim, stoch=stoch)
    for sp, index in enumerate([0, 10, 20, 39]):
        plt.subplot(2, 2, sp + 1)
        for st in range(steps):
            plt.plot(
                st * np.ones_like(xf[:, index, st]),
                xf[:, index, st],
                ".",
                alpha=0.1,
                color="grey",
            )
            plt.plot(st, np.mean(xf[:, index, st], 0), "o", color="black")

            plt.plot(
                st * np.ones_like(xa[:, index, st]),
                xa[:, index, st],
                ".",
                alpha=0.1,
                color="blue",
            )
            plt.plot(st, np.mean(xa[:, index, st], 0), "o", color="blue")

            plt.plot(st, y[index, st], "o", color="red")

    plt.show()
    # plt.subplot(1, 2, 1)
    # plt.contourf((xa.mean(0)))
    # plt.subplot(1, 2, 2)
    # plt.contourf(y)
    # plt.show()


def EnKF_oscillator(steps, N, stoch, freqobs=50):
    xf = np.empty((2, N, steps, freqobs))
    xa = np.empty((2, N, steps))
    obs = np.empty(steps)
    H = np.atleast_2d([0, 1])
    R = np.atleast_2d([49])
    # Observational covariance error matrix

    oscillator = NonLinearOscillatorModel_2d()
    oscillator.initial_state([0, 1])
    oscillator.step(freqobs)
    y = H @ oscillator.state_vector[:, -1]
    iter_step = 0
    prior_covariance = np.asarray(
        [
            [5, 0],
            [0, 5],
        ]
    )
    xf_i = scipy.random.multivariate_normal(
        mean=np.asarray([1, 1]), cov=prior_covariance, size=N
    ).T

    obs[iter_step] = y
    for i in range(freqobs):
        xf_i, xf_bar, Pf = forecast_oscillator(xf_i)
        xf[:, :, iter_step, i] = xf_i

    xa_i, xa_bar, Pa = analysis(xf_i, y, H, np.atleast_2d(Pf), R, N, stoch)
    xa[:, :, iter_step] = xa_i
    iter_step = 1
    while iter_step < steps:
        print(iter_step)
        xf_i = xa_i
        for i in range(freqobs):
            xf_i, xf_bar, Pf = forecast_oscillator(xf_i, 1)
            xf[:, :, iter_step, i] = xf_i

        oscillator.step(freqobs)
        y = H @ oscillator.state_vector[:, -1] + np.random.randn(1) * np.sqrt(
            49
        )  # srqt(49)
        xa_i, xa_bar, Pa = analysis(xf_i, y, H, np.atleast_2d(Pf), R, N, stoch)
        xa[:, :, iter_step] = xa_i
        obs[iter_step] = y
        iter_step += 1
    return xf, xa, obs, oscillator


def main_nonlinear_oscillator(
    assimilation_steps=100, N=200, period_assim=50, stoch=True
):

    # Set parameters of the EnKF
    H = np.atleast_2d([0, 1])
    R = np.atleast_2d([256])

    EnKF_o = EnKF(
        state_dimension=2, ensemble_size=N, period_assim=period_assim, H=H, R=R
    )
    # Use prior information to generate ensemble
    EnKF_o.generate_ensemble(np.array([0, 1]), np.asarray([[10, 2], [2, 10]]))

    def forward(x):
        oscil = NonLinearOscillatorModel_2d()
        oscil.initial_state(x)
        oscil.step(1)
        return oscil.state_vector[:, -1]

    oscillator = NonLinearOscillatorModel_2d()
    oscillator.initial_state([0, 1])

    def generate_observations():
        oscillator.step(EnKF_o.period_assim)
        y = oscillator.state_vector[:, -1] + np.random.randn(1) * np.sqrt(256)
        return EnKF_o.H @ y

    EnKF_o.get_observation = generate_observations
    EnKF_o.forwardmodel = forward
    EnKF_o.assimilation(assimilation_steps)

    ## Exploiting and visualizing results
    xf_bar = EnKF_o.xf_ensemble_total.mean(1)
    xf_std = EnKF_o.xf_ensemble_total.std(1)

    plt.plot(xf_bar[1, :])
    plt.fill_between(
        x=np.arange(len(xf_bar[1, :])),
        y1=xf_bar[1, :] + xf_std[1, :],
        y2=xf_bar[1, :] - xf_std[1, :],
        color="gray",
        alpha=0.4,
    )
    obs_x = np.arange(
        EnKF_o.period_assim,
        EnKF_o.period_assim * (assimilation_steps + 1),
        EnKF_o.period_assim,
    )
    # plt.plot(obs_x, xa_bar[1, :], color="magenta")
    plt.plot(obs_x, np.asarray(EnKF_o.observations), "o", color="red")
    plt.plot(oscillator.state_vector[1, :], ":")
    plt.show()


if __name__ == "__main__":
    # main_lorenz40(steps=20, N=500, dim=40, stoch=True)

    main_nonlinear_oscillator(assimilation_steps=10, N=200, period_assim=100)
