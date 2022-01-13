import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Callable, Tuple
import scipy.linalg as la
from .baseparticlefilter import BaseParticleFilter

# from lorenz import LorenzModel
# from dynamicalsystems.anharmonic_oscillator import NonLinearOscillatorModel


def unscented_transform(
    mean: np.ndarray,
    cov: np.ndarray,
    L: int,
    alpha: float = 1e-3,
    beta: float = 2,
    kappa: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    Lambda = alpha ** 2 * (L + kappa) - L
    Pfsqrt = la.cholesky((Lambda + L) * cov)
    sigma_points = np.empty((len(mean), 2 * L + 1))
    weights = np.empty(2 * L + 1)
    weights[1:] = 1 / (2 * (L + Lambda))
    weights[0] = Lambda / (L + Lambda)
    sigma_points[:, 0] = mean
    print(f"{Pfsqrt.shape=}")
    print(f"{mean.shape=}")

    for i in range(L):
        sigma_points[:, i + 1] = mean + Pfsqrt[i, :]
        sigma_points[:, L + i + 1] = mean - Pfsqrt[i, :]
    return sigma_points, weights


def unscented_mean_covariance(
    sigma_points: np.ndarray,
    weights: np.ndarray,
    alpha: float = 1e-3,
    beta: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.average(sigma_points, weights=weights, axis=1)
    weights_c = weights
    weights_c[0] += 1 - alpha ** 2 + beta
    cov = np.cov(sigma_points, aweights=weights_c)
    return mean, cov


class UnscentedPF(ParticleFilter):
    def generate_particles(
        self, prior_mean: np.ndarray, prior_cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.particles, self.weights = unscented_transform(prior_mean, prior_cov)
