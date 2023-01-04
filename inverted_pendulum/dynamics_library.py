import numpy as np
import types
import inspect
from copy import copy


class RCVeCarContinuous:
    def __init__(self, alpha=-0.4, beta=1.2, gamma= 8):
        # x = [x, y, theta, v] and u = [throttle, steer]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, x, u):
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        if len(u.shape) == 1:
            u = u.reshape(u.shape[0], 1)
        assert x.shape[0] == 4
        assert u.shape[0] == 2
        assert x.shape[1] == u.shape[1]
        n_points = x.shape[1]
        x_dot = np.zeros(x.shape)
        x_dot[0, :] = x[3, :] * np.cos(x[2, :])
        x_dot[1, :] = x[3, :] * np.sin(x[2, :])
        x_dot[2, :] = x[3, :] * self.alpha * u[1, :]
        x_dot[3, :] = -self.beta + self.gamma*u[0, :]


class PendulumCartContinuous:
    def __init__(self, m=0.2, g=9.8, l=0.3, j=0.006, b=0.1):
        self.m = m
        self.g = g
        self.l = l
        self.j = j
        self.b = b

    def __call__(self, x: np.ndarray, u: np.ndarray):
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        if len(u.shape) == 1:
            u = u.reshape(u.shape[0], 1)
        assert x.shape[0] == 2
        assert u.shape[0] == 1
        assert x.shape[1] == u.shape[1]
        n_points = x.shape[1]
        x_dot = np.zeros(x.shape)
        m, g, l, j, b = self.m, self.g, self.l, self.j, self.b
        jt = j + m*l*l
        x_dot[0, :] = x[1, :]
        x_dot[1, :] = m * g * l / jt * np.sin(x[0, :]) - b / jt * x[1, :] + l / jt * np.cos(x[0, :]) * u[0, :]
        return x_dot


class PendulumCartBroken:
    def __init__(self, m=0.2, g=9.8, l=0.3, j=0.006, b=0.1):
        self.m = m
        self.g = g
        self.l = l
        self.j = j
        self.b = b

    def __call__(self, x: np.ndarray, u: np.ndarray):
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        if len(u.shape) == 1:
            u = u.reshape(u.shape[0], 1)
        assert x.shape[0] == 2
        assert u.shape[0] == 1
        assert x.shape[1] == u.shape[1]
        n_points = x.shape[1]
        x_dot = np.zeros(x.shape)
        m, g, l, j, b = self.m, self.g, self.l, self.j, self.b

        b_vec = b*np.ones((1, x.shape[1]))
        b_vec[0, (0.5 <= x[0, :]) & (x[0, :] <= 2.0)] = 4.0 * b
        b_vec[0, (-2.0 <= x[0, :]) & (x[0, :] <= -0.5)] = 2.0 * b

        jt = j + m*l*l
        x_dot[0, :] = x[1, :]
        x_dot[1, :] = m * g * l / jt * np.sin(x[0, :]) - b_vec / jt * x[1, :] + l / jt * np.cos(x[0, :]) * u[0, :]
        return x_dot


class DoubleIntegrator:
    def __init__(self):
        self.i = 1

    def __call__(self, x, u):
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        if len(u.shape) == 1:
            u = u.reshape(u.shape[0], 1)
        assert x.shape[0] == 2
        assert u.shape[0] == 1
        assert x.shape[1] == u.shape[1]
        n_points = x.shape[1]
        x_dot = np.zeros(x.shape)
        x_dot[0, :] = x[1, :]
        x_dot[1, :] = u * self.i
        return x_dot


class SampleAndHold:
    def __init__(self, continuous_function: types.FunctionType, sample_time: float, discretization_step=-1, backward=False):
        assert inspect.isclass(type(continuous_function)) or isinstance(continuous_function, types.FunctionType)
        self.continuous_function = continuous_function
        self.sample_time = sample_time
        self.backward = backward
        if discretization_step == -1:
            self.discretization_step = sample_time
        else:
            self.discretization_step = discretization_step

    def __call__(self, x: np.ndarray, u: np.ndarray, return_path=False):
        x_hist = [x]
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        if len(u.shape) == 1:
            u = u.reshape(u.shape[0], 1)
        t = 0
        while t < self.sample_time:
            x_dot = self.continuous_function(x, u)
            if self.backward:
                x_dot = -x_dot
                
            x_plus = x + x_dot * self.discretization_step
            x_hist.append(copy(x_plus))
            t += self.discretization_step
            x = x_plus
        if return_path:
            return x_plus, x_hist
        else:
            return x_plus