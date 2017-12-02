import math
import unittest
import numpy as np
import hopfield_np as hop

from src import input


class TestHopfieldNumpy(unittest.TestCase):
    def test_inputs_init(self):
        data = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        net = hop.HopfieldNet(data, 1, 0)
        for x in range(0, len(data)):
            for y in range(0, len(data)):
                self.assertAlmostEqual(net.inputs[x][y], 1 / 9, 2)

    def test_activations_from_array(self):
        data = np.array([
            [-1, 0, 3],
            [3, -1, 0],
            [0, 0, 0]
        ])
        net = hop.HopfieldNet(data, 1, 0)
        acts = net.activation(data[0, :])
        self.assertAlmostEqual(acts[0], 0, 1)
        self.assertAlmostEqual(acts[1], 0.5, 1)
        self.assertAlmostEqual(acts[2], 1, 1)

        acts = net.activation(data[1, :])
        self.assertAlmostEqual(acts[0], 1, 1)
        self.assertAlmostEqual(acts[1], 0, 1)
        self.assertAlmostEqual(acts[2], 0.5, 1)

        acts = net.activation(data[:, 0])
        self.assertAlmostEqual(acts[0], 0, 1)
        self.assertAlmostEqual(acts[1], 1, 1)
        self.assertAlmostEqual(acts[2], 0.5, 1)

    def test_get_sum_of_activations(self):
        data = np.array([
            [-1, 0, 3],
            [3, -1, 0],
            [0, 0, 0]
        ])
        net = hop.HopfieldNet(data, 1, 0)
        self.assertEqual(np.sum(net.activation(data[0, :])), 1.5, 6)
        self.assertEqual(np.sum(net.activation(data[1, :])), 1.5, 6)
        self.assertEqual(np.sum(net.activation(data[:, 0])), 1.5, 6)

    def test_get_a(self):
        data = np.array([
            [-1, 0, 3],
            [3, -1, 0],
            [0, 0, 0]
        ])
        net = hop.HopfieldNet(data, 1, 0)
        b = net.activation(net.inputs[0, 1])
        c = net.activation(net.inputs[0, 2])
        self.assertAlmostEqual(net.get_a(0, 0), (b + c) * net.a, 6)

    def test_get_b(self):
        data = np.array([
            [-0.75, -0.25, 0],
            [0.01, 0.1, 1],
            [-0.1, 0, 0.1]
        ])
        net = hop.HopfieldNet(data, 1, 0)
        b = net.activation(net.inputs[0, 0])
        c = net.activation(net.inputs[2, 0])
        self.assertAlmostEqual(net.get_b(1, 0), (b + c) * net.b, 6)

    def test_get_c(self):
        data = np.array([
            [-0.75, -0.25, 0],
            [0.01, 0.1, 1],
            [-0.1, 0, 0.1]
        ])
        net = hop.HopfieldNet(data, 1, 0)
        manual_sum = 0.0
        manual_sum += net.activation(net.inputs[0, 0])
        manual_sum += net.activation(net.inputs[0, 1])
        manual_sum += net.activation(net.inputs[0, 2])
        manual_sum += net.activation(net.inputs[1, 0])
        manual_sum += net.activation(net.inputs[1, 1])
        manual_sum += net.activation(net.inputs[1, 2])
        manual_sum += net.activation(net.inputs[2, 0])
        manual_sum += net.activation(net.inputs[2, 1])
        manual_sum += net.activation(net.inputs[2, 2])
        manual_sum -= net.size + net.size_adj
        manual_sum *= net.c
        self.assertAlmostEqual(net.get_c(), manual_sum, 6)


    def test_get_d(self):
        data = np.array([
            [0, 0.25, 0.8],
            [0.25, 0, 1],
            [0.8, 1, 0]
        ])
        net = hop.HopfieldNet(data, 1, 0)
        city1 = (net.activation(net.inputs[0, 0]) + net.activation(net.inputs[0, 2])) * net.distances[
            1, 0]
        city2 = (net.activation(net.inputs[1, 0]) + net.activation(net.inputs[1, 2])) * net.distances[
            1, 1]
        city3 = (net.activation(net.inputs[2, 0]) + net.activation(net.inputs[2, 2])) * net.distances[
            1, 2]

        manual_sum = (city1 + city2 + city3) * net.d
        self.assertEqual(net.get_d(1, 1), manual_sum )


if __name__ == '__main__':
    unittest.main()
