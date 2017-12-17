import os
import asyncio
import concurrent.futures
import logging
import sys

from running.args_parse import get_args
from running.run import Run, RunParams

from hopfield.input import read_data
from running.paths import Paths
from storage.data_storage import *

async def run_series(params_list, stores_list, root_path):
    log = logging.getLogger("run_series")
    executor = concurrent.futures.ProcessPoolExecutor()
    event_loop = asyncio.get_event_loop()
    runs = [Run(params_list[i], stores_list[i], root_path) for i in range(0, len(params_list))]
    run_tasks = [event_loop.run_in_executor(executor, runs[i].begin)
                 for i in range(0, len(runs))]
    log.info(f"Preparing {len(run_tasks)} runs to run")
    await asyncio.wait(run_tasks)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='PID %(process)5s %(name)18s: %(message)s',
        stream=sys.stderr
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Root at: " + root)
    paths = Paths(root)
    data = read_data(fr"{paths.input()}\burma14.txt")
    args = get_args()
    data_store = DataStorage(paths.results(), args.tag)
    event_loop = asyncio.get_event_loop()
    series_params = [RunParams(seed, args.steps, size_adj, data, args.freq,
                               args.freq, args.tag, args.images, args.video)
                     for seed in args.seeds for size_adj in args.size_adjs]
    run_stores = [data_store.open_run_store(run_index)
                  for run_index in range(len(args.seeds) * len(args.size_adjs))]
    event_loop.run_until_complete(run_series(series_params, run_stores, root))


if __name__ == '__main__':
    main()
