import matplotlib.pyplot as plt


class Plotter:
    def __init__(self):
        self.fig = plt.figure(figsize=(30, 10), dpi=50)
        self.subplot = 1

    def add_subplot(self, points, cmap, vmin, vmax, title):
        self.fig.add_subplot(1, 3, self.subplot)
        self.subplot += 1

        plt.imshow(points, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        plt.title(title)
        plt.colorbar()

    def plot(self, title, path):
        plt.suptitle(title)
        plt.savefig(path)
        plt.close()
