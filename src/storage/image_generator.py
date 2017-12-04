import os
import subprocess as sp
import sys
from pathlib import Path

from storage.plotter import Plotter


class GraphicalGenerator:
    def __init__(self, runStore, ffmpeg_path):
        self.runStore = runStore
        self.ffmpeg_path = ffmpeg_path

    def generate_run_video(self, params):
        print("\nCreating video with ffmpeg")
        ffmpeg_command = f"{self.ffmpeg_path} -loglevel panic -r 10 -i {self.runStore.get_image_path_regexp()} " \
                         f"-vframes {int(params.steps/params.freq)} {self.runStore.get_video_path()}"
        sp.call(ffmpeg_command, stdout=open(os.devnull, 'wb'))
        video_file = Path(self.runStore.get_video_path())
        if video_file.is_file():
            print(f"Video file created at: '{video_file}'")
        else:
            print("No video created :(")

    def generate_run_images(self, params, cords, distances):
        print("Generating images!")
        for index in range(0, len(self.runStore)):
            self.plot_data_point(index,
                                 params.freq,
                                 cords,
                                 distances)
            sys.stdout.write(f"Image {index} out of {int(params.steps / params.freq)}\r")
        print("\nIt is done")

    def plot_data_point(self, index, freq, cords, distances):
        net_conf = self.runStore.get_net_config()
        net_state = self.runStore.get_data_point(index)
        plotter = Plotter(6)
        plotter.add_subplot(net_state["activations"], 'hot', 0, 1, f"Activations")
        plotter.add_subplot(net_state["inputs"], 'coolwarm', -0.075, 0.075, f"Outputs of neurons")
        plotter.add_subplot(net_state["inputsChange"], 'Blues_r', -0.001, 0, f"Negative change")
        plotter.add_subplot(net_state["inputsChange"], 'Reds', 0, 0.001, f"Positive change")
        plotter.add_subplot(distances, 'plasma', 0, 1, f"Distance matrix")
        plotter.add_graph(self.get_map(net_state["activations"], cords))
        plotter.plot(
            f"a {net_conf['a']}; b {net_conf['b']}; c {net_conf['c']}; "
            f"d {net_conf['d']}; size_adj"
            f" {net_conf['size_adj']}; u0 {net_conf['u0']}; "
            f"timestep {net_conf['timestep']}; Step: {index * freq}",
            f"{self.runStore.get_image_path(index)}")

    def get_map(self, acts, cords):
        points = []
        for pos in range(0, len(acts)):
            for city in range(0, len(acts)):
                if acts[city][pos] > 0.6:
                    points.append(cords[city])

        return points
