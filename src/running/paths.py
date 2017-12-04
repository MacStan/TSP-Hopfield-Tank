import os
from pathlib import Path


class Paths:
    def __init__(self, root):
        self._root = root
        ensure_existance(self.results())
        ensure_existance(self.input())

    def results(self):
        return rf"{self._root}\data\results"

    def input(self):
        return rf"{self._root}\data\input"


def ensure_existance(raw_path):
    path = Path(raw_path)
    if not path.is_dir():
        print(rf"Creating '{path}'")
        os.makedirs(raw_path)
