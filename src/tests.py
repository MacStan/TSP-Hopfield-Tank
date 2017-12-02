import math
import unittest
import subprocess as sp
import hopfield
import os, sys
from src import input
from runner import main
from argparse import Namespace
import shutil

class TestInput(unittest.TestCase):
    def test_three_points(self):
        matrix = input.distance_matrix([(0, 0), (0, 2), (4, 0)])
        self.assertAlmostEqual(matrix[0][0], 0.0)
        self.assertAlmostEqual(matrix[0][1], 2.0)
        self.assertAlmostEqual(matrix[0][2], 4.0)

        self.assertAlmostEqual(matrix[1][0], 2.0)
        self.assertAlmostEqual(matrix[1][1], 0.0)
        self.assertAlmostEqual(matrix[1][2], 2 * math.sqrt(5), 2)

        self.assertAlmostEqual(matrix[2][0], 4.0)
        self.assertAlmostEqual(matrix[2][1], 2 * math.sqrt(5), 2)
        self.assertAlmostEqual(matrix[2][2], 0.0)

    def test_four_points(self):
        matrix = input.distance_matrix([(1, 3), (2, 1), (4, 4), (6, 2)])

        self.assertAlmostEqual(matrix[0][0], 0.0)
        self.assertAlmostEqual(matrix[0][1], 2.23, 1)
        self.assertAlmostEqual(matrix[0][2], 3.16, 1)
        self.assertAlmostEqual(matrix[0][3], 5.09, 1)

        self.assertAlmostEqual(matrix[1][0], 2.23, 1)
        self.assertAlmostEqual(matrix[1][1], 0.0, 1)
        self.assertAlmostEqual(matrix[1][2], 3.60, 1)
        self.assertAlmostEqual(matrix[1][3], 4.12, 1)

        self.assertAlmostEqual(matrix[2][0], 3.16, 1)
        self.assertAlmostEqual(matrix[2][1], 3.60, 1)
        self.assertAlmostEqual(matrix[2][2], 0.0, 1)
        self.assertAlmostEqual(matrix[2][3], 2.82, 1)

        self.assertAlmostEqual(matrix[3][0], 5.09, 1)
        self.assertAlmostEqual(matrix[3][1], 4.12, 1)
        self.assertAlmostEqual(matrix[3][2], 2.82, 1)
        self.assertAlmostEqual(matrix[3][3], 0.0, 1)

    def test_normalisation(self):
        matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 10],
        ]
        matrix = input.normalize(matrix)
        self.assertAlmostEqual(matrix[0][0], 0.1)
        self.assertAlmostEqual(matrix[0][1], 0.2)
        self.assertAlmostEqual(matrix[0][2], 0.3)

        self.assertAlmostEqual(matrix[1][0], 0.4)
        self.assertAlmostEqual(matrix[1][1], 0.5)
        self.assertAlmostEqual(matrix[1][2], 0.6)

        self.assertAlmostEqual(matrix[2][0], 0.7)
        self.assertAlmostEqual(matrix[2][1], 0.8)
        self.assertAlmostEqual(matrix[2][2], 1.0)


class TestHopfield(unittest.TestCase):
    def test_inputs_init(self):
        matrix = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        net = hopfield.HopfieldNet(matrix, 1, 0)
        for x in range(0, len(matrix)):
            for y in range(0, len(matrix)):
                self.assertAlmostEqual(net.inputs[x][y], 1 / 9, 3)

    def test_functional(self):
        os.chdir(".//src//")
        main(True, Namespace(seeds=[1], size_adjs=[0], steps=50, freq=10, tag='UNITTEST'))
        dirs = os.listdir("..//plots")
        valid_dirs = [x for x in dirs if "UNITTEST" in x]
        for dir in valid_dirs:
            contents = os.listdir(f"..//plots//{dir}")
            self.assertTrue("img4.png" in contents)
            self.assertTrue("run.mp4" in contents)
        for dir in valid_dirs:
            shutil.rmtree(f"..//plots//{dir}")


if __name__ == '__main__':
    unittest.main()
