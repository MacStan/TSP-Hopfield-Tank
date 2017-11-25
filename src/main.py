from plotter import Plotter
from hopfield import HopfieldNet
from input import distance_matrix, normalize, read_data
import subprocess as sp
import datetime as dt
import os
import time

burma14 = read_data("./input_data/burma14.txt")
distances = distance_matrix(burma14)
normalized_distances = normalize(distances)

for seed in range(0, 5):
    net = HopfieldNet(normalized_distances, seed)
    step = 0
    img = 0
    freq = 20
    steps = 5000
    date = dt.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
    new_path = f"..\\plots\\{date}-seed{seed}-steps{steps}\\"
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    old = time.time()
    print(time.time() - old)

    for step in range(0, steps):
        print(f"time {time.time() - old}")
        old = time.time()

        step += 1
        net.update()
        print(f"Update iter: {step}")

        if step % freq == 0:
            acts = net.activations()
            plotter = Plotter()

            plotter.add_subplot(acts, 'hot', 0, 1, f"Activations {img * freq}")
            plotter.add_subplot(net.inputs, 'coolwarm', -0.05, 0.05, f"Inputs {img * freq}")
            plotter.add_subplot(net.inputsChange, 'bwr', -0.001, 0.001,
                                f"Inputs changes {img * freq}")

            plotter.plot(
                f"seed 1; a {net.a}; b {net.b}; c {net.c}; d {net.d}; size_adj"
                f" {net.size_adj}; u0 {net.u0}; timestep {net.timestep};",
                f"{new_path}\img{img}.png")

            img += 1

    ffmpeg_command = f"ffmpeg -r 10 -i {new_path}img%d.png -vframes {steps/freq} {new_path}run.mp4"
    print(ffmpeg_command)
    sp.call(ffmpeg_command)
    open(f"{new_path}\Success", "w")
