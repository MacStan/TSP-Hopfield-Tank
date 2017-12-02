import datetime as dt
import os
import subprocess as sp
import sys
import time
from pathlib import Path
from hopfield import HopfieldNet
from input import distance_matrix, normalize, normalize_cords
from image_generator import ImageGenerator


class RunParams:
    def __init__(self, seed, steps, size_adj, data, freq, tag, plot,video ):
        self.seed = seed
        self.steps = steps
        self.size_adj = size_adj
        self.data = data
        self.freq = freq
        self.tag = tag
        self.do_plot = plot
        self.do_video = video


def run(params: RunParams, run_store):
    create_logs_path()
    net, new_path, normalized_distances = initialize(params)
    run_store.store_net_config(net.get_net_configuration())

    print("\nAnnealing network")
    optimize_network(run_store, params.freq, net, params.steps)
    print("\nAnnealing done!\n")

    if (params.do_plot):
        ImageGenerator(new_path).generate_run_images(
            params, normalize_cords(params.data), normalized_distances, run_store)

    print("\nCreating video with ffmpeg")
    ffmpeg_command = f"ffmpeg -loglevel panic -r 10 -i {new_path}img%d.png " \
                     f"-vframes {int(params.steps/params.freq)} {new_path}run.mp4"

    if (params.do_video):
        sp.call(ffmpeg_command, stdout=open(os.devnull, 'wb'))

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
    net = HopfieldNet(normalized_distances, params.seed, params.size_adj)
    date = dt.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    path = create_plots_path(date, params.tag, params.seed, params.steps)

    return net, path, normalized_distances


def create_plots_path(date, tag, seed, steps):
    path = f"..\\plots\\{date}-{str(tag)}-seed{seed}-steps{steps}\\"
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created log directory: {path}")
    return path


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


def create_logs_path():
    path = f"..\\logs"
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created log directory: {path}")
    return path
