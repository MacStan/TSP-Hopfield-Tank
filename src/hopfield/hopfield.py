from math import tanh
import random
import logging as lg
import datetime as dt
import os

class HopfieldNet:
    def __init__(self, matrix, seed, size_adj):

        # values taken from paper
        self.inputsChange = []
        self.a = 500
        self.b = 500
        self.c = 200
        self.d = 500

        self.u0 = 0.02
        self.timestep = 0.000001
        self.distances = matrix
        self.size = len(matrix)
        self.size_adj = size_adj

        self.seed = seed
        self.inputs = self.init_inputs()
        if not os.path.exists("./logs"):
            os.makedirs("./logs")
        self.logger = lg.getLogger('HopfieldNet')
        lg.basicConfig(
            filename=f'./logs/example-{str(dt.datetime.now().strftime("%Y%m%d-%H%M%S"))}.log',
            level=lg.INFO)

    def init_inputs(self):
        random.seed(self.seed)
        base = 1 / (self.size ** 2)
        init = []
        for x in range(0, self.size):
            row = []
            for y in range(0, self.size):
                row.append(base + ((random.random()-0.5) / 10000))
            init.append(row)
        return init

    def activation(self, input):
        sigm = 0.5 * (1 + tanh(input / self.u0))
        activ = 0 if sigm < 0.2 else sigm
        activ = 1 if sigm > 0.8 else activ
        return sigm

    def get_a(self, city, position):
        sum = 0.0
        for pos in range(0, self.size):
            sum += self.activation(self.inputs[city][pos])
        sum -= self.activation(self.inputs[city][position])
        return sum * self.a

    def get_b(self, mainCity, position):
        sum = 0.0
        for city in range(0, self.size):
            sum += self.activation(self.inputs[city][position])
        sum -= self.activation(self.inputs[mainCity][position])
        return sum * self.b

    def get_c(self):
        sum = 0.0
        for city in range(0, self.size):
            for pos in range(0, self.size):
                sum += self.activation(self.inputs[city][pos])
        sum -= self.size + self.size_adj
        return sum * self.c

    def get_d(self, main_city, position):
        sum = 0.0
        for city in range(0, self.size):
            preceding = self.activation(self.inputs[city][(position + 1) % self.size])
            following = self.activation(self.inputs[city][(position - 1)])
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
        self.inputsChange = []

        for city in range(0, self.size):
            row = []
            for pos in range(0, self.size):
                row.append(self.timestep * self.get_states_change(city, pos))
            self.inputsChange.append(row)

        for city in range(0, self.size):
            for pos in range(0, self.size):
                self.inputs[city][pos] += self.inputsChange[city][pos]
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

    def activations_printable(self):
        activations = []
        for x in range(0, self.size):
            row = []
            for y in range(0, self.size):
                act = self.activation(self.inputs[x][y])
                sign = "X" if act > 0.75 else "_"
                row.append(sign)
            activations.append(f"{x}# " + " ".join(row))

        return "\n".join(activations)

    def inputs_printable(self):
        activations = []
        for x in range(0, self.size):
            row = []
            for y in range(0, self.size):
                row.append(str(f"{self.inputs[x][y]:.2f}"))
            activations.append(" ".join(row))

        return "\n".join(activations)

    def encoded_path_valid(self):
        valid = True
        for x in range(0, self.size):
            counter = 0.0
            for y in range(0, self.size):
                counter += self.activation(self.inputs[x][y])
            valid &= self.activations_vector_validity("x", x, counter)
        for y in range(0, self.size):
            counter = 0.0
            for x in range(0, self.size):
                counter += self.activation(self.inputs[x][y])
            valid &= self.activations_vector_validity("y", y, counter)

        if valid:
            self.logger.info("SUCCESS")
        else:
            self.logger.info("FAIL")

    def activations_vector_validity(self, cord, cord_pos, counter):
        if counter <= 0.1:
            self.logger.debug(f"FAIL, sum less or equa1 0.1. {cord}:{cord_pos} sum: {counter}")
        if counter > 1.1:
            self.logger.debug(f"FAIL, sum greater than 1 by 0.1. {cord}:{cord_pos} sum: {counter}")
        if (not (counter <= 0.1)) and (not (counter > 1.1)):
            return True
        else:
            return False

    def get_net_configuration(self):
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d, "u0": self.u0,
                "size_adj": self.size_adj, "timestep": self.timestep}

    def get_net_state(self):
        return {"activations": self.activations(), "inputs": self.inputs,
                "inputsChange": self.inputsChange}
