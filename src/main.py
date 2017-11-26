from plotter import Plotter
from hopfield import HopfieldNet
from input import distance_matrix, normalize, read_data
from args_parse import get_args
from data_storage import DataStorage
import subprocess as sp
import datetime as dt
import os
import time

def generate_images(new_path, seedRange, dataPointRange, freq):
    print("Generating images!")
    for seedIndex in range(0, seedRange):
        for dataPointIndex in range(0, dataPointRange):
            data_point = dataStorage.get_data_point(seedIndex, dataPointIndex)
            plot_data_point(new_path, dataStorage.get_net_configuration(seedIndex), dataStorage.get_data_point(seedIndex, dataPointIndex), dataPointIndex, freq)
    print("It is done")

def plot_data_point(new_path, netConfiguration, netState, imgIndex, freq):
    plotter = Plotter()
    plotter.add_subplot(netState["activations"], 'hot', 0, 1, f"Activations {imgIndex * freq}")
    plotter.add_subplot(netState["inputs"], 'coolwarm', -0.05, 0.05, f"Inputs {imgIndex * freq}")
    plotter.add_subplot(netState["inputsChange"], 'bwr', -0.001, 0.001, f"Inputs changes {imgIndex * freq}")
    plotter.plot(
        f"seed 1; a {netConfiguration['a']}; b {netConfiguration['b']}; c {netConfiguration['c']}; d {netConfiguration['d']}; size_adj"
        f" {netConfiguration['size_adj']}; u0 {netConfiguration['u0']}; timestep {netConfiguration['timestep']};",
        f"{new_path}\img{imgIndex}.png")


burma14 = read_data("./input_data/burma14.txt")
distances = distance_matrix(burma14)
normalized_distances = normalize(distances)
args = get_args()
dataStorage = DataStorage()


for seed in range(0, args.seed):
    net = HopfieldNet(normalized_distances, seed)
    step = 0
    img = 0
    date = dt.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    new_path = f"..\\plots\\{date}-seed{seed}-steps{args.steps}\\"
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    old = time.time()
    print(time.time() - old)
    dataStorage.start_new_seed(seed, net.get_net_configuration())

    for step in range(0, args.steps):
        print(f"time {time.time() - old}")
        old = time.time()
        net.update()
        step += 1
        print(f"Update iter: {step}")

        if step % args.freq == 0:
            dataStorage.save_data_point(net.get_net_state(), img)
            img += 1

    generate_images(new_path, args.seed, int(args.steps/args.freq), args.freq)
    ffmpeg_command = f"ffmpeg -r 10 -i {new_path}img%d.png -vframes {args.steps/args.freq} {new_path}run.mp4"
    print(ffmpeg_command)
    sp.call(ffmpeg_command)
    open(f"{new_path}\Success", "w")
