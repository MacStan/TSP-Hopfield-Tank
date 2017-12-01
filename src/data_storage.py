import json
import os
import datetime as dt
import sys
from plotter import Plotter


class DataStorage:
    def __init__(self, dataStoragePath=f"../data"):
        runTimestamp = dt.datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
        self.runStoragePath = os.path.join(dataStoragePath, f"{runTimestamp}")
        if not os.path.exists(self.runStoragePath):
            os.makedirs(self.runStoragePath)

    def start_new_seed(self, seedIndex, netConfiguration):
        self.currentSeedStoragePath = os.path.join(self.runStoragePath, f"seed{seedIndex}")
        if not os.path.exists(self.currentSeedStoragePath):
            os.makedirs(self.currentSeedStoragePath)
        with open(os.path.join(self.currentSeedStoragePath, f"net-configuration"),
                  'x') as configFile:
            configFile.write(json.dumps(netConfiguration))

    def save_data_point(self, netState, dataPointIndex):
        with open(os.path.join(self.currentSeedStoragePath, f"data-point{dataPointIndex}"),
                  'x') as dataFile:
            dataFile.write(json.dumps(netState))

    def get_net_configuration(self, seedIndex):
        with open(os.path.join(self.runStoragePath,
                               f"seed{seedIndex}/net-configuration")) as configFile:
            return json.loads(configFile.read())

    def get_data_point(self, seedIndex, dataPointIndex):
        with open(os.path.join(self.runStoragePath, f"seed{seedIndex}/data-point{dataPointIndex}"),
                  'r') as dataFile:
            return json.loads(dataFile.read())

    def generate_images(self, new_path, seed, dataPointRange, freq, cords, distances):
        print("Generating images!")

        for dataPointIndex in range(0, dataPointRange):
            self.plot_data_point(new_path,
                                 self.get_net_configuration(seed),
                                 self.get_data_point(seed, dataPointIndex),
                                 dataPointIndex,
                                 freq,
                                 cords,
                                 distances)
            sys.stdout.write(f"Image {dataPointIndex} out of {dataPointRange}\r")
        print("\nIt is done")

    def get_map(self, acts, cords):
        points = []
        for pos in range(0, len(acts)):
            for city in range(0, len(acts)):
                if acts[city][pos] > 0.6:
                    points.append(cords[city])

        return points

    def plot_data_point(self, new_path, net_conf, net_state, img_index, freq, cords, distances):
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
            f"{new_path}\img{img_index}.png")
