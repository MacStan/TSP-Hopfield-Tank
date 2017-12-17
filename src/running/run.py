import sys
import time
import os
import logging

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

class Run:
    def __init__(self, run_params, run_store, root_path):
        self.run_params = run_params
        self.run_store = run_store
        self.root_path = root_path

    def begin(self):
        net, normalized_distances = self.initialize()
        self.run_store.store_net_config(net.get_net_configuration())
        graphical_generator = GraphicalGenerator(self.run_store,
                                                os.path.join(self.root_path, "ffmpeg"))

        print("Annealing network")
        self.optimize_network(net)
        print("Annealing done!")

        if self.run_params.do_images or self.run_params.do_video:
            graphical_generator.generate_run_images(self.run_params, normalize_cords(self.run_params.data),
                                                    normalized_distances)
        if self.run_params.do_video:
            graphical_generator.generate_run_video(self.run_params)

        print("Run Ended\n")

    def initialize(self):
        print(f"Seed: {self.run_params.seed}; Steps: {self.run_params.steps}; "
            f"Size_Adj: {self.run_params.size_adj}; Freq: {self.run_params.freq}")
        normalized_distances = normalize(distance_matrix(self.run_params.data))
        net = HopfieldNet(normalized_distances, self.run_params.seed, self.run_params.size_adj, self.run_params.paths)

        return net, normalized_distances

    def optimize_network(self, net):
        old = time.time()
        for step in range(0, self.run_params.steps):
            aligned_step = '{:>5}'.format(step)
            sys.stdout.write(f"Step: {aligned_step} Time: {time.time() - old:.2}\r")
            old = time.time()
            net.update()

            if step % self.run_params.freq == 0:
                self.run_store.add_data_point(net.get_net_state())
        self.run_store.commit_data()
