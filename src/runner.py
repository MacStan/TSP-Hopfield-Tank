import traceback
from multiprocessing import Pool
from main import run, RunParams
from input import read_data
from args_parse import get_args
from data_storage import *
import os, re
from pathlib import Path
def run_wrapper(argList):
    run(argList[0], argList[1])

def main(flag, args ):

    if not flag:
        args = get_args()
    data = read_data("./input_data/burma14.txt")
    processInputArgs = []
    dataStore = DataStorage()
    runIndex = 0

    for seed in args.seeds:
        for size_adj in args.size_adjs:
            processInputArgs.append([
                RunParams(
                    seed, args.steps, size_adj, data, args.freq, args.tag, args.plot, args.video),
                dataStore.open_run_store(runIndex)])
            runIndex += 1

    pool = Pool()
    pool.map(run_wrapper, processInputArgs)

if __name__ == '__main__':
    main()