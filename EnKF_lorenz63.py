#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from dynamicalsystems.lorenz63 import Lorenz63Model
from DAmethod.EnKF import EnKF, Kalman_gain


def forecast_lorenz63(x_ensemble):
    N = x_ensemble.shape[0]
    xf_i = np.empty((N, 3))
    for i, x in enumerate(x_ensemble):
        lo = Lorenz63Model()
        lo.initial_state(x)
        lo.step(0.02, 10)
        xf_i[i, :] = lo.history[1][:, -1]
    xf_bar = xf_i.mean(0)
    Pf = np.cov(xf_i.T)
    return xf_i, xf_bar, Pf


def forward(x, period_assim):
    lor = Lorenz63Model()
    lor.initial_state(x)
    lor.step(dt, period_assim)
    return lor.history[1][:, -1]


if __name__ == "__main__":
    H = np.asarray([[1, 0, 0]])
    R = np.array([[0.2]])

    lorenz_truth = Lorenz63Model()
    lorenz_truth.initial_state(np.array([0, 1, 0]))
    dt = 0.02
    # Burn in
    lorenz_truth.step(dt, 1000)
    initial_state = lorenz_truth.history[1][:, -1] + np.random.normal(
        0, np.sqrt(0.2), size=3
    )

    EnKF_63 = EnKF(state_dimension=3, ensemble_size=20,
                   period_assim=20, H=H, R=R)
    EnKF_63.generate_ensemble(
        initial_state,
        np.asarray(
            [
                [0.1, 0, 0],
                [0, 0.1, 0],
                [0, 0, 0.1],
            ]
        ),
    )

    def get_obs():
        lorenz_truth.step(dt, EnKF_63.period_assim)
        print(lorenz_truth.history[1][:, -1])
        obs = H @ lorenz_truth.history[1][:, -1] + np.random.normal(
            loc=0, scale=np.sqrt(0.2), size=(1)
        )
        return obs

    EnKF_63.get_observation = get_obs
    EnKF_63.forwardmodel = lambda x: forward(x, EnKF_63.period_assim)

    Nsteps = 100

    EnKF_63.assimilation(Nsteps)

    xf_bar = EnKF_63.xf_ensemble_total.mean(1)
    xf_std = EnKF_63.xf_ensemble_total.std(1)

    xa_bar = EnKF_63.xa_ensemble_total.mean(1)
    xa_std = EnKF_63.xa_ensemble_total.std(1)

    obs_x = np.arange(
        EnKF_63.period_assim,
        EnKF_63.period_assim * (100 + 1),
        EnKF_63.period_assim,
    )

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(obs_x, xa_bar[i, :])
        plt.plot(xf_bar[i, :])
        # plt.fill_between(
        #     x=np.arange(len(xa_bar[i, :])),
        #     y1=xa_bar[i, :] + xa_std[i, :],
        #     y2=xa_bar[i, :] - xa_std[i, :],
        #     color="gray",
        #     alpha=0.4,
        # )
        if i == 0:
            plt.plot(obs_x, np.asarray(EnKF_63.observations), "o", color="red")

    plt.show()

    def truth():
        lorenz_truth = Lorenz63Model()
        lorenz_truth.initial_state(np.array([0, 1, 0]))
        dt = 0.02
        # Burn in
        lorenz_truth.step(dt, (EnKF_63.period_assim * Nsteps))
        return lorenz_truth.history[1]

    y = truth()
    plt.plot(obs_x, np.asarray(EnKF_63.observations), "o", color="red")
    plt.plot(y[0, :])
    plt.show()
