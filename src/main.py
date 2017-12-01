import datetime as dt
import os
import subprocess as sp
import sys
import time
from pathlib import Path

from data_storage import DataStorage
from hopfield import HopfieldNet
from input import distance_matrix, normalize, normalize_cords


class RunParams:
    def __init__(self, seed, steps, size_adj, data, freq, tag):
        self.seed = seed
        self.steps = steps
        self.size_adj = size_adj
        self.data = data
        self.freq = freq
        self.tag = tag


def run(params: RunParams):
    data_storage, net, new_path, normalized_distances = initialize(params)

    data_storage.start_new_seed(params.seed, net.get_net_configuration())

    print("\nAnnealing network")
    optimize_network(data_storage, params.freq, net, params.steps)
    print("\nAnnealing done!\n")

    data_storage.generate_images(new_path, params.seed, int(params.steps / params.freq),
                                 params.freq,
                                 normalize_cords(params.data),
                                 normalized_distances)

    print("\nCreating video with ffmpeg")
    ffmpeg_command = f"ffmpeg -loglevel panic -r 10 -i {new_path}img%d.png -vframes {int(params.steps/params.freq)} " \
                     f"{new_path}run.mp4"

    sp.call(ffmpeg_command, stdout=open(os.devnull, 'wb'))
    open(f"{new_path}\Success", "w", )

    my_file = Path(f"{new_path}run.mp4")
    if my_file.is_file():
        print(f"Video file created at: '{my_file}'")
    else:
        print("No video created :(")
    print("Run Ended\n")


def initialize(params):
    print(
        f"Seed: {params.seed}; Steps: {params.steps}; Size_Adj: {params.size_adj}; Freq: {params.freq}")
    normalized_distances = normalize(distance_matrix(params.data))
    data_storage = DataStorage()

    net = HopfieldNet(normalized_distances, params.seed, params.size_adj)
    date = dt.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    path = create_plots_path(date, params.tag, params.seed, params.steps)
    return data_storage, net, path, normalized_distances


def create_plots_path(date, tag, seed, steps):
    path = f"..\\plots\\{date}-{str(tag)}-seed{seed}-steps{steps}\\"
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created log directory: {path}")
    return path


def optimize_network(data_storage, freq, net, steps):
    old = time.time()
    for step in range(0, steps):
        aligned_step = '{:>5}'.format(step)
        sys.stdout.write(f"Step: {aligned_step} Time: {time.time() - old:.2}\r")
        old = time.time()
        net.update()

        if step % freq == 0:
            data_storage.save_data_point(net.get_net_state(), int(step / freq))
