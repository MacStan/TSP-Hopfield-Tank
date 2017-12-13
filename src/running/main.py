import sys
import time
import os

from hopfield.hopfield_np import HopfieldNet
from hopfield.input import distance_matrix, normalize, normalize_cords
from running.paths import Paths
from storage.image_generator import GraphicalGenerator


class RunParams:
    def __init__(self, seed, steps, size_adj, data, freq, tag, do_images, do_video, paths: Paths):
        self.seed = seed
        self.steps = steps
        self.size_adj = size_adj
        self.data = data
        self.freq = freq
        self.tag = tag
        self.do_images = do_images
        self.do_video = do_video
        self.paths = paths


def run(params: RunParams, run_store, root_path):
    net, normalized_distances = initialize(params)
    run_store.store_net_config(net.get_net_configuration())
    graphical_generator = GraphicalGenerator(run_store,
                                             os.path.join(root_path, "ffmpeg"))

    print("\nAnnealing network")
    optimize_network(run_store, params.freq, net, params.steps)
    print("\nAnnealing done!\n")

    if params.do_images or params.do_video:
        graphical_generator.generate_run_images(params, normalize_cords(params.data),
                                                normalized_distances)
    if params.do_video:
        graphical_generator.generate_run_video(params)

    print("Run Ended\n")


def initialize(params):
    print(f"Seed: {params.seed}; Steps: {params.steps}; "
          f"Size_Adj: {params.size_adj}; Freq: {params.freq}")
    normalized_distances = normalize(distance_matrix(params.data))
    net = HopfieldNet(normalized_distances, params.seed, params.size_adj, params.paths)

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
