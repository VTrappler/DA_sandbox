import numpy as np
import matplotlib.pyplot as plt
from solvers import integrate_RK4
import cmasher as cmr

Nx = 36
Nu = 360


def phi(v, pm):
    return np.roll(v, pm) * (np.roll(v, -pm) - np.roll(v, 2 * pm))


def xdot(x, u, F=15, h=1, c=10, b=10):
    return (
        phi(x, 1)
        + F
        - (h * c / b) * np.fromiter(map(sum, np.split(u, Nx)), dtype="float")
    )


def udot(x, u, h=1, c=10, b=10):
    return (c / b) * phi(b * u, -1) + (h * c / b) * x[np.arange(len(u)) // 10]


def lorenz2scale_dot(t, xu):
    c = 10  # Time scale ratio
    b = 10  # Space scale ratio
    h = 1  # Coupling
    F = 15  # Forcing
    Nx = 36
    x, u = xu[:Nx], xu[Nx:]
    dx = xdot(x, u, F=F, h=h, c=c, b=b)
    du = udot(x, u, h=h, c=c, b=b)
    return np.concatenate([dx, du])


class Lorenz2scalesModel:
    def __init__(self, Nx=36):
        "docstring"
        self.Nx = Nx
        self.Nu = 10 * Nx
        self.dim = self.Nx + self.Nu

    @staticmethod
    def dotfunction(t, xu):
        return lorenz2scale_dot(t, xu)

    def integrate():
        pass


if __name__ == "__main__":
    initial_state = np.random.normal(0, 1, size=(Nx + Nu))
    dt = 0.0001
    t = 0

    Nsteps = 100_000
    t, x = integrate_RK4(lorenz2scale_dot, 0, initial_state, dt, Nsteps)

    initial_state[380] += 0.0001
    t, x2 = integrate_RK4(lorenz2scale_dot, 0, initial_state, dt, Nsteps)

    plt.imshow(x - x2, aspect="auto", cmap="cmr.iceburn")
    plt.show()
