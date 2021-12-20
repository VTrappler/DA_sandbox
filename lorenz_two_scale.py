import numpy as np
import scipy
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    initial_state = np.random.normal(0, 1, size=(Nx + Nu))
    dt = 0.005
    inte = scipy.integrate.solve_ivp(
        lorenz2scale_dot,
        y0=initial_state,
        t_span=(0, 10),
    )
    plt.subplot(1, 2, 1)
    plt.imshow(inte.y[:Nx], aspect="auto")
    plt.subplot(1, 2, 2)
    plt.imshow(inte.y[Nx:], aspect="auto")
    plt.show()
