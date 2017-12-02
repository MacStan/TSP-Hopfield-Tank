from multiprocessing import Pool

from args_parse import get_args
from hopfield.input import read_data
from main import run, run_params
from storage.data_storage import *


def run_wrapper(arg_list):
    run(arg_list[0], arg_list[1])


def main():
    data = read_data(f"./input_data/burma14.txt")
    args = get_args()
    process_input_args = []
    data_store = DataStorage()

    run_index = 0
    for seed in args.seeds:
        for size_adj in args.size_adjs:
            process_input_args.append([
                run_params(
                    seed, args.steps, size_adj, data, args.freq, args.tag, args.images, args.video),
                data_store.open_run_store(run_index)])
            run_index += 1

    pool = Pool()
    pool.map(run_wrapper, process_input_args)


if __name__ == '__main__':
    main()
