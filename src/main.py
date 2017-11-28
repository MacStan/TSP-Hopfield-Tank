from plotter import Plotter
from hopfield import HopfieldNet
from data_storage import DataStorage
from input import distance_matrix, normalize, normalize_cords
from pathlib import Path
import subprocess as sp
import datetime as dt
import os
import time
import sys


def generate_images(new_path, seed, dataPointRange, freq, cords, dataStorage, distances):
    print("Generating images!")

    for dataPointIndex in range(0, dataPointRange):
        plot_data_point(new_path,
                        dataStorage.get_net_configuration(seed),
                        dataStorage.get_data_point(seed, dataPointIndex),
                        dataPointIndex,
                        freq,
                        cords,
                        distances)
        sys.stdout.write(f"Image {dataPointIndex} out of {dataPointRange}\r")
    print("\nIt is done")


def get_map(acts, cords):
    points = []
    for pos in range(0, len(acts)):
        for city in range(0, len(acts)):
            if (acts[city][pos] > 0.6):
                points.append(cords[city])
                
    return points


def plot_data_point(new_path, net_conf, net_state, imgIndex, freq, cords, distances):
    plotter = Plotter(6)
    plotter.add_subplot(net_state["activations"], 'hot', 0, 1, f"Activations")
    plotter.add_subplot(net_state["inputs"], 'coolwarm', -0.075, 0.075, f"Outputs of each neuron")
    plotter.add_subplot(net_state["inputsChange"], 'Blues_r', -0.001, 0, f"Negative change")
    plotter.add_subplot(net_state["inputsChange"], 'Reds', 0, 0.001, f"Positive change")
    plotter.add_subplot(distances, 'plasma', 0, 1, f"Distance matrix")
    plotter.add_graph(get_map(net_state["activations"], cords))
    plotter.plot(
        f"a {net_conf['a']}; b {net_conf['b']}; c {net_conf['c']}; "
        f"d {net_conf['d']}; size_adj"
        f" {net_conf['size_adj']}; u0 {net_conf['u0']}; "
        f"timestep {net_conf['timestep']}; Step: {imgIndex * freq}",
        f"{new_path}\img{imgIndex}.png")


def run(seed, steps, size_adj, data, freq, tag):
    print(f"Seed: {seed}; Steps: {steps}; Size_Adj: {size_adj}; Freq: {freq}")
    distances = distance_matrix(data)
    normalized_distances = normalize(distances)
    normalized_cords = normalize_cords(data)

    data_storage = DataStorage()

    net = HopfieldNet(normalized_distances, seed, size_adj)

    date = dt.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    new_path = f"..\\plots\\{date}-{str(tag)}-seed{seed}-steps{steps}\\"
    if not os.path.exists(new_path):
        os.makedirs(new_path)
        print(f"Created log directory: {new_path}")
    old = time.time()
    data_storage.start_new_seed(seed, net.get_net_configuration())
    step = 0
    img = 0
    print("\nAnnealing network")
    for step in range(0, steps):
        formated_step = '{:>5}'.format(step)
        sys.stdout.write(f"Step: {formated_step} Time: {time.time() - old:.2}\r")
        old = time.time()
        net.update()
        step += 1

        if step % freq == 0:
            data_storage.save_data_point(net.get_net_state(), img)
            img += 1
    print("\nAnnealing done!")
    print()
    generate_images(new_path, seed, int(steps / freq), freq, normalized_cords, data_storage, normalized_distances )
    print()
    print("Creating video with ffmpeg")
    ffmpeg_command = f"ffmpeg -loglevel panic -r 10 -i {new_path}img%d.png -vframes {int(steps/freq)} " \
                     f"{new_path}run.mp4"

    sp.call(ffmpeg_command,stdout=open(os.devnull, 'wb'))
    open(f"{new_path}\Success", "w", )

    my_file = Path(f"{new_path}run.mp4")
    if my_file.is_file():
        print(f"Video file created at: '{my_file}'")
    else:
        print("No video created :(")
    print("Run Ended")
    print()
