from math import tanh
import random
import logging as lg
import datetime as dt
import numpy as np


class HopfieldNet:
    def __init__(self, distances, seed, size_adj):

        # values taken from paper
        self.size = len(distances)

        self.inputs_change = np.zeros([self.size, self.size], float)
        self.a = 500
        self.b = 500
        self.c = 200
        self.d = 500

        self.u0 = 0.02
        self.timestep = 0.000001
        self.distances = distances

        self.size_adj = size_adj

        self.seed = seed
        self.inputs = self.init_inputs()
        self.logger = lg.getLogger('HopfieldNet')
        lg.basicConfig(
            filename=f'../logs/example-{str(dt.datetime.now().strftime("%Y%m%d-%H%M%S"))}.log',
            level=lg.INFO)

    def init_inputs(self):
        random.seed(self.seed)
        base = np.ones([self.size, self.size], float)
        base /= self.size ** 2
        for x in range(0, self.size):
            for y in range(0, self.size):
                base[x][y] += ((random.random() - 0.5) / 10000)
        return base

    def activation(self, single_input):
        sigm = 0.5 * (1 + np.tanh(single_input / self.u0))
        return sigm

    def get_a(self, city, position):
        sum = np.sum(self.activation(self.inputs[city, :]))
        sum -= self.activation(self.inputs[city, position])
        return sum * self.a

    def get_b(self, main_city, position):
        sum = np.sum(self.activation(self.inputs[:, position]))
        sum -= self.activation(self.inputs[main_city][position])
        return sum * self.b

    def get_c(self):
        sum = np.sum(self.activation(self.inputs[:, :]))
        sum -= self.size + self.size_adj
        return sum * self.c

    def get_d(self, main_city, position):
        sum = 0.0
        for city in range(0, self.size):
            preceding = self.activation(self.inputs[city, (position + 1) % self.size])
            following = self.activation(self.inputs[city, (position - 1)])
            sum += self.distances[main_city][city] * (preceding + following)

        return sum * self.d

    def get_states_change(self, city, pos):
        new_state = -self.inputs[city][pos]
        new_state -= self.get_a(city, pos)
        new_state -= self.get_b(city, pos)
        new_state -= self.get_c()
        new_state -= self.get_d(city, pos)
        return new_state

    def update(self):
        self.inputs_change = np.zeros([self.size, self.size], float)

        for city in range(0, self.size):
            for pos in range(0, self.size):
                self.inputs_change[city, pos] = self.timestep * self.get_states_change(city, pos)

        self.inputs += self.inputs_change
        pass

    def activations(self):
        activations = []
        for x in range(0, self.size):
            row = []
            for y in range(0, self.size):
                act = self.activation(self.inputs[x][y])
                row.append(act)
            activations.append(row)
        return activations




    def get_net_configuration(self):
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d, "u0": self.u0,
                "size_adj": self.size_adj, "timestep": self.timestep}

    def get_net_state(self):
        return {"activations": self.activations(), "inputs": self.inputs.tolist(),
                "inputsChange": self.inputs_change.tolist()}
