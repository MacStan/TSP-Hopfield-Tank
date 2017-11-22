from src.hopfield import HopfieldNet
import matplotlib.pyplot as plt
from src.input import distance_matrix, normalize
import subprocess as sp
import datetime as dt
import os
import time
def compare_array(a, b, size):
    for x in range(0, size):
        for y in range(0, size):
            if b[x][y] != a[x][y]:
                return False
    return True


burma14 = [
    # (16.47, 96.10),
    # (16.47, 94.44),
    # (20.09, 92.54),
    # (22.39, 93.37),
    (25.23, 97.24),
    (22.00, 96.05),
    (20.47, 97.02),
    (17.20, 96.29),
    (16.30, 97.38),
    (14.05, 98.12),
    (16.53, 97.38),
    (21.52, 95.59),
    (19.41, 97.13),
    (20.09, 94.55)]

distances = distance_matrix(burma14)
normalized_distances = normalize(distances)

net = HopfieldNet(normalized_distances)
step = 0
img = 0
freq = 10
date = dt.datetime.now().strftime("%S-%M-%H_%d-%m-%Y")
new_path = f".\\plots\\{date}\\"
if not os.path.exists(new_path):
    os.makedirs(new_path)

old = time.time()
print(time.time() - old)
for step in range(0, 10000):
    print(time.time() - old)
    old = time.time()

    step += 1
    net.update()
    print(f"Update iter: {step}")

    if step % freq == 0:
        acts = net.activations()

        fig = plt.figure(figsize=(30,10))
        a = fig.add_subplot(1, 3, 1)
        plt.imshow(acts, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
        plt.title(f"Activations {img * freq}")
        plt.colorbar()

        a = fig.add_subplot(1, 3, 2)
        plt.imshow(net.inputs, cmap='coolwarm',vmin=-0.5, vmax=0.5, interpolation='nearest')
        plt.title(f"Inputs {img * freq}")
        plt.colorbar()

        a = fig.add_subplot(1, 3, 3)
        plt.imshow(net.statesChange, cmap='coolwarm',vmin=-0.001, vmax=0.001,interpolation='nearest')
        plt.title(f"Inputs changes {img * freq}")
        plt.colorbar()


        plt.savefig(f"{new_path}\img{img}.png")
        plt.close()

        img += 1

ffmpeg_command = f"ffmpeg -r 10 -i {new_path}img%d.png -vframes 2000 {new_path}run.mp4"
print(ffmpeg_command)
sp.call(ffmpeg_command)
open(f"{new_path}\Success","w")


