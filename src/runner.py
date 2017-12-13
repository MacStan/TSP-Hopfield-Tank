import os
from multiprocessing import Pool

from running.args_parse import get_args
from running.main import run, RunParams

from hopfield.input import read_data
from running.paths import Paths
from storage.data_storage import *


def run_wrapper(arg_list):
    run(arg_list[0], arg_list[1], arg_list[2])


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Root at: " + root)
    paths = Paths(root)
    data = read_data(fr"{paths.input()}\burma14.txt")
    args = get_args()
    process_input_args = []
    data_store = DataStorage(paths.results(), args.tag)

    run_index = 0
    for seed in args.seeds:
        for size_adj in args.size_adjs:
            process_input_args.append([
                RunParams(
                    seed, args.steps, size_adj, data, args.freq, args.tag, args.images, args.video,
                    paths),
                data_store.open_run_store(run_index),
                root])
            run_index += 1

    pool = Pool()
    pool.map(run_wrapper, process_input_args)


if __name__ == '__main__':
    main()
