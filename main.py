import hopfield
from input import distance_matrix, normalize
from hopfield import HopfieldNet

cords = [(0.4000, 0.4439),
         (0.2439, 0.1463),
         (0.1707, 0.2293),
         (0.2293, 0.7610),
         (0.5171, 0.9414),
         (0.8732, 0.6536),
         (0.6878, 0.5219),
         (0.8488, 0.3609),
         (0.6683, 0.2536),
         (0.6195, 0.2634)]

distances = distance_matrix(cords)
normalized_distances = normalize(distances)

net = HopfieldNet(normalized_distances)

for step in range(0, 10000):
    net.update()
    if step % 100 == 0:
        print(step)
        path = net.activations_printable()
        print( path)
        print("## 0 1 2 3 4 5 6 7 8 9 ")
        net.encoded_path_valid()
        print()
        print(net.inputs_printable() + "\n\n")

