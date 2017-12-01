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
            [-1,  0, 3],
            [ 3, -1, 0],
            [ 0,  0, 0]
        ])
        net = hop.HopfieldNet(data, 1, 0)
        acts = net.activation(data[0,:])
        self.assertAlmostEqual(acts[0], 0, 1)
        self.assertAlmostEqual(acts[1], 0.5, 1)
        self.assertAlmostEqual(acts[2], 1, 1)

        acts = net.activation(data[1,:])
        self.assertAlmostEqual(acts[0], 1, 1)
        self.assertAlmostEqual(acts[1], 0, 1)
        self.assertAlmostEqual(acts[2], 0.5, 1)

        acts = net.activation(data[:,0])
        self.assertAlmostEqual(acts[0], 0, 1)
        self.assertAlmostEqual(acts[1], 1, 1)
        self.assertAlmostEqual(acts[2], 0.5, 1)

if __name__ == '__main__':
    unittest.main()
