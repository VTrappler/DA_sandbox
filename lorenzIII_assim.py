#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from lorenzIII import Lorenz2scalesModel
from lorenz93 import Lorenz93Model


Nx, Nu = 36, 360
initial_state = np.random.normal(0, 1, size=(Nx + Nu))
dt = 0.001
Lorenz2scalesModel.dt = dt
lorenzIII_t = Lorenz2scalesModel()
lorenzIII_t.set_initial_state(0, initial_state)
lorenzIII_t.forward(2000)

y0 = np.copy(lorenzIII_t.state_vector[:, -1])

Nk = 50  # steps between observations


def observation_operator(model, state_vector: np.ndarray) -> np.ndarray:
    """H"""
    return state_vector[: model.Nx, -1]


def obserr(model) -> np.ndarray:
    """vk"""
    return np.random.multivariate_normal(np.zeros(model.Nx), cov=5 * np.eye(model.Nx))


Lorenz2scalesModel.set_observation_operator(observation_operator)
Lorenz2scalesModel.set_observation_errors(obserr)

y0 = lorenzIII_t.state_vector[:, -1] + 0.0001 * np.random.normal(
    0, 1, size=(Nx + Nu)
)  # Perturbate initial condition
t0 = lorenzIII_t.t[-1]
t_obs = [t0]
y = [y0[:Nx]]
xs = y0
x = np.empty((36, 0))
t = np.empty(0)
for i in tqdm.trange(100, desc="Simulations"):
    lorenzIII_t.forward(Nk)
    y_ = lorenzIII_t.observe()
    y.append(y_)

    t_, x_ = Lorenz2scalesModel.integrate(t_obs[-1], xs, Nk, verbose=False)
    xs = x_[:, -1]
    x = np.concatenate([x, x_[:Nx]], axis=1)
    t = np.concatenate([t, t_])
    t_obs.append(t_[-1])


y = np.asarray(y).T
plt.plot(t, x[0, :])
plt.plot(t_obs, y[0, :], "ro")
plt.plot(lorenzIII_t.t[2000:], lorenzIII_t.state_vector[0, 2000:])
plt.show()

## Ensemble members:
Ne = 64
x0 = lorenzIII_t.state_vector[:, -1]

ensemble_members = [Lorenz2scalesModel() for _ in range(Ne)]

cov = np.diag(np.append(np.zeros(Nx), np.ones(Nu)))

for ens in ensemble_members:
    ens.set_initial_state(
        0, x0 + np.random.multivariate_normal(np.zeros(Nx + Nu), cov=cov)
    )


def ensemble_forward(ensemble, Nk):
    for i, xi in enumerate(ensemble):
        xi.forward(Nk)


ensemble_forward(ensemble_members, 100)
lorenzIII_t.forward(Nk)

for ens in ensemble_members:
    plt.plot(ens.state_vector[0, :], color="gray", alpha=0.2)

plt.plot(lorenzIII_t.state_vector[0, -50:])
plt.show()


H = np.hstack([np.eye(Nx), np.zeros((Nx, Nu))])
y


def Kalman_gain(H, Pf, R):
    return np.linalg.multi_dot(
        [
            Pf,
            H.T,
            np.linalg.inv(np.linalg.multi_dot([H, Pf, H.T]) + R),
        ]
    )


for en in ensemble_members:
    d = H @ y0 - H @ en.state_vector[:, -1]
    xa = H @ en.state_vector[:, -1] + 0.5 * d
