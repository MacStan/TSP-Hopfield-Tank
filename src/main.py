from src.hopfield import HopfieldNet
import matplotlib.pyplot as plt
from src.input import distance_matrix, normalize


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
for step in range(0, 2000):
    step += 1
    net.update()

    if step % freq == 0:
        acts = net.activations()
        plt.imshow(acts, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
        plt.title(f"{img * freq}")
        plt.savefig(f"./plots/img{img}.png")
        img += 1
