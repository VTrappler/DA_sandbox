#!/usr/bin/env python
# coding: utf-8
import numpy as np
from tqdm import tqdm
from functools import partial


def RK4_step(f, t, x, dt, *args):
    """Apply a step from the Runge-Kutta of order 4 method with fixed stepsize
    x' = f(t,x)

    :param f: function which defines the ODE
    :param t: time parameter
    :param x: state vector
    :param dt: stepsize
    :returns: update state vector

    """
    k1 = f(t, x, *args)
    k2 = f(t + dt / 2, x + k1 * dt / 2.0, *args)
    k3 = f(t + dt / 2, x + k2 * dt / 2.0, *args)
    k4 = f(t + dt, x + k3 * dt, *args)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def DoPri45_step(f, t, x, h):
    """Apply a step using Dormand-Prince method

    :param f: function which defines the ODE
    :param t: time
    :param x: state vector
    :param h: stepsize
    :returns: update state vector

    """
    k1 = f(t, x)
    k2 = f(t + 1.0 / 5 * h, x + h * (1.0 / 5 * k1))
    k3 = f(t + 3.0 / 10 * h, x + h * (3.0 / 40 * k1 + 9.0 / 40 * k2))
    k4 = f(
        t + 4.0 / 5 * h,
        x + h * (44.0 / 45 * k1 - 56.0 / 15 * k2 + (32.0 / 9) * k3),
    )
    k5 = f(
        t + 8.0 / 9 * h,
        x
        + h
        * (
            19372.0 / 6561 * k1
            - 25360.0 / 2187 * k2
            + 64448.0 / 6561 * k3
            - 212.0 / 729 * k4
        ),
    )
    k6 = f(
        t + h,
        x
        + h
        * (
            9017.0 / 3168 * k1
            - 355.0 / 33 * k2
            + 46732.0 / 5247 * k3
            + 49.0 / 176 * k4
            - 5103.0 / 18656 * k5
        ),
    )

    v5 = (
        35.0 / 384 * k1
        + 500.0 / 1113 * k3
        + 125.0 / 192 * k4
        - 2187.0 / 6784 * k5
        + 11.0 / 84 * k6
    )
    k7 = f(t + h, x + h * v5)
    v4 = (
        5179.0 / 57600 * k1
        + 7571.0 / 16695 * k3
        + 393.0 / 640 * k4
        - 92097.0 / 339200 * k5
        + 187.0 / 2100 * k6
        + 1.0 / 40 * k7
    )

    return x + h * v5


def integrate_step(step, f, t0, x0, dt, Nsteps, *args):
    t = np.empty(Nsteps + 1)
    t[0] = t0
    t_ = t0
    x = np.empty((len(x0), Nsteps + 1))
    x[:, 0] = x0
    curr_x = x0
    for i in tqdm.trange(Nsteps):
        curr_x = step(f, t, curr_x, dt, *args)
        t_ += dt
        t[i + 1] = t_
        x[:, i + 1] = curr_x
    return t, x


def integrate_RK4(f, t0, x0, dt, Nsteps, *args):
    """Integrate the given ODE for Nsteps

    :param f:
    :param t0:
    :param x0:
    :param dt:
    :param Nsteps:
    :returns:

    """
    return integrate_step(RK4_step, f, t0, x0, dt, Nsteps, *args)
