from math import tanh
import random

class HopfieldNet:
    def __init__(self, matrix):

        # values taken from paper
        self.a = 500
        self.b = 500
        self.c = 200
        self.d = 500

        self.u0 = 0.02
        self.timestep = 0.000001
        self.distances = matrix
        self.size = len(matrix)

        self.inputs = self.init_inputs()

    def init_inputs(self):
        init = []
        for x in range(0, self.size):
            row = []
            for y in range(0, self.size):
                row.append( (1 / self.size ** 2)
                            + random.uniform(-0.1,0.1)
                            )
            init.append(row)
        return init

    def activation(self, input):
        return 0.5 * (1 + tanh(input / self.u0))
        # activ = 0 if sigm < 0.2 else sigm
        # activ = 1 if sigm > 0.8 else activ
        # return sigm;

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
                self.activation(self.inputs[city][pos])
        sum -= self.size
        return sum * self.c

    def get_d(self, mainCity, position):
        sum = 0.0
        for city in range(0, self.size):
            minPos = self.size - 1 if position - 1 < 0else position - 1
            maxPos = 0 if position + 1 >= self.size else position + 1
            sum += self.distances[mainCity][city] \
                   * (self.activation(self.inputs[city][minPos])
                      + self.activation(self.inputs[city][maxPos]))

        return sum * self.d

    def get_new_state(self, city, pos):
        newState = -self.inputs[city][pos]
        newState -= self.get_a(city, pos)
        newState -= self.get_b(city, pos)
        newState -= self.get_c()
        newState -= self.get_d(pos, city)
        return newState

    def update(self):
        statesChange = []

        for city in range(0, self.size):
            row = []
            for pos in range(0, self.size):
                row.append(self.timestep * self.get_new_state(city, pos))
            statesChange.append(row)

        for city in range(0, self.size):
            for pos in range(0, self.size):
                self.inputs[city][pos] += statesChange[city][pos]

    def activations_printable(self):
        activations = []
        for x in range(0, self.size):
            row = []
            for y in range(0, self.size):
                row.append("{0:.1f}".format(self.activation(self.inputs[x][y])))
            activations.append(" ".join(row))

        return "\n".join(activations)
