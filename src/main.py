import copy

from HopfieldRaw import HopfieldNet

from src.input import distance_matrix, normalize


def compare_array(a, b, size):
    for x in range(0, size):
        for y in range(0, size):
            if b[x][y] != a[x][y]:
                return False
    return True


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

net = HopfieldNet(normalized_distances)
step = 0
while True:
    step += 1
    last_states = copy.deepcopy(net.inputs)
    net.update()
    if compare_array(last_states, net.inputs, len(last_states)):
        break

    if step % 100 == 0:
        print(step)
        path = net.activations_printable()
        print(path)
        print("## 0 1 2 3 4 5 6 7 8 9 ")
        net.encoded_path_valid()
        print()
        print(net.inputs_printable() + "\n\n")
