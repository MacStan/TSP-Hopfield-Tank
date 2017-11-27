from math import sqrt
from typing import List, Tuple


def read_data(file):
    stream = open(file)
    lines = stream.readlines()

    coordinates = []
    for line in lines:
        parts = line.split(" ")
        coordinates.append((float(parts[0]), float(parts[1])))
    return coordinates


def distance(p1, p2):
    return sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))


def distance_matrix(coordinates: List[Tuple]):
    matrix = []
    for cord1 in coordinates:
        row = []
        for cord2 in coordinates:
            row.append(distance(cord1, cord2))
        matrix.append(row)
    return matrix


def get_largest(matrix):
    largest = 0.0
    for x in range(0, len(matrix)):
        largest = largest if largest > max(matrix[x]) else max(matrix[x])
    return largest


def normalize(matrix):
    largest = get_largest(matrix)

    for x in range(0, len(matrix)):
        for y in range(0, len(matrix)):
            matrix[x][y] /= largest
    return matrix


def normalize_cords(matrix):
    xs = []
    ys = []
    for point in matrix:
        xs.append(point[0])
        ys.append(point[1])
    largest_x = max(xs)
    largest_y = max(ys)

    for pos in range(0, len(matrix)):
        xs[pos] /= largest_x
        ys[pos] /= largest_y

    return list(zip(xs, ys))
