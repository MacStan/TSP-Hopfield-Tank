import datetime as dt
import os
import sys
import time

from data_storage import DataStorage
from hopfield import HopfieldNet
from input import distance_matrix, normalize, normalize_cords
from image_generator import ImageGenerator


class RunParams:
    def __init__(self, seed, steps, size_adj, data, freq, tag):
        self.seed = seed
        self.steps = steps
        self.size_adj = size_adj
        self.data = data
        self.freq = freq
        self.tag = tag


def run(params: RunParams, runStore):
    net, normalized_distances = initialize(params)
    runStore.store_net_config(net.get_net_configuration())
    imageGenerator = ImageGenerator(runStore)

    print("\nAnnealing network")
    optimize_network(runStore, params.freq, net, params.steps)
    print("\nAnnealing done!\n")

    imageGenerator.generate_run_images(
        params, normalize_cords(params.data), normalized_distances)
    imageGenerator.generate_run_video(params)

    print("Run Ended\n")


def initialize(params):
    print(
        f"Seed: {params.seed}; Steps: {params.steps}; Size_Adj: {params.size_adj}; Freq: {params.freq}")
    normalized_distances = normalize(distance_matrix(params.data))
    net = HopfieldNet(normalized_distances, params.seed, params.size_adj)

    return net, normalized_distances


def optimize_network(runStore, freq, net, steps):
    old = time.time()
    for step in range(0, steps):
        aligned_step = '{:>5}'.format(step)
        sys.stdout.write(f"Step: {aligned_step} Time: {time.time() - old:.2}\r")
        old = time.time()
        net.update()

        if step % freq == 0:
            runStore.add_data_point(net.get_net_state())
    runStore.commit_data()
