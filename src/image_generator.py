import sys
from plotter import Plotter

class ImageGenerator():
    def __init__(self, new_path):
        self.new_path = new_path

    def generate_run_images(self, params, cords, distances, runStore):
        print("Generating images!")

        for dataPointIndex in range(0, len(runStore)):
            self.plot_data_point(runStore.get_net_config(),
                                 runStore.get_data_point(dataPointIndex),
                                 dataPointIndex,
                                 params.freq,
                                 cords,
                                 distances)
            sys.stdout.write(f"Image {dataPointIndex} out of {int(params.steps / params.freq)}\r")
        print("\nIt is done")

    def plot_data_point(self, net_conf, net_state, img_index, freq, cords, distances):
        plotter = Plotter(6)
        plotter.add_subplot(net_state["activations"], 'hot', 0, 1, f"Activations")
        plotter.add_subplot(net_state["inputs"], 'coolwarm', -0.075, 0.075,
                            f"Outputs of each neuron")
        plotter.add_subplot(net_state["inputsChange"], 'Blues_r', -0.001, 0, f"Negative change")
        plotter.add_subplot(net_state["inputsChange"], 'Reds', 0, 0.001, f"Positive change")
        plotter.add_subplot(distances, 'plasma', 0, 1, f"Distance matrix")
        plotter.add_graph(self.get_map(net_state["activations"], cords))
        plotter.plot(
            f"a {net_conf['a']}; b {net_conf['b']}; c {net_conf['c']}; "
            f"d {net_conf['d']}; size_adj"
            f" {net_conf['size_adj']}; u0 {net_conf['u0']}; "
            f"timestep {net_conf['timestep']}; Step: {img_index * freq}",
            f"{self.new_path}\img{img_index}.png")

    def get_map(self, acts, cords):
        points = []
        for pos in range(0, len(acts)):
            for city in range(0, len(acts)):
                if acts[city][pos] > 0.6:
                    points.append(cords[city])

        return points
