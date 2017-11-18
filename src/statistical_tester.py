import hopfield as hf
from src.input import distance_matrix, normalize

burma14 = [
    #(16.47, 96.10),
    #(16.47, 94.44),
    #(20.09, 92.54),
    #(22.39, 93.37),
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

net = hf.HopfieldNet(normalized_distances)

for step in range(0, 10000):
    net.update()
    net.encoded_path_valid()
    print(net.activations_printable())

