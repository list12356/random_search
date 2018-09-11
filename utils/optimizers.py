import numpy as np


class Optimizer(object):
    def __init__(self, model):
        self.model = model
        self.num_param = len(model.theta_G)
        self.t = 0

    def update(self, update_list):
        assert len(update_list) == len(self.model.theta_G), "incorrect number of parameters"
        self.t += 1
        step = self._compute_step(update_list)
        # theta = self.model.theta_G
        # ratio = np.linalg.norm(step) / np.linalg.norm(theta)
        self.model.update(step)
        # return ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, model, stepsize, momentum=0.9):
        Optimizer.__init__(self, model)
        self.v = [np.zeros(shape=size, dtype=np.float32) for size in self.model.size]
        self.stepsize, self.momentum = stepsize, momentum

    def _compute_step(self, update_list):
        for t in range(len(update_list)):
            self.v[t] = self.momentum * self.v[t] + (1. - self.momentum) * update_list[t]
            update_list[t] = self.stepsize * self.v[t]
        return update_list


class Adam(Optimizer):
    def __init__(self, model, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        Optimizer.__init__(self, model)
        self.stepsize = stepsize
        self.model = model
        self.num_param = len(model.theta_G)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = [np.zeros(shape=size, dtype=np.float32) for size in self.model.size]
        self.v = [np.zeros(shape=size, dtype=np.float32) for size in self.model.size]

    def _compute_step(self, update_list):
        a = self.stepsize * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        for t in range(len(update_list)):
            self.m[t] = self.beta1 * self.m[t] + (1 - self.beta1) * update_list[t]
            self.v[t] = self.beta2 * self.v[t] + (1 - self.beta2) * (update_list[t] * update_list[t])
            update_list[t] = a * self.m[t] / (np.sqrt(self.v[t]) + self.epsilon)
        return update_list
