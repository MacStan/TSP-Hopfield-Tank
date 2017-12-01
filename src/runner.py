import traceback
from main import run, RunParams
from input import read_data
from args_parse import get_args

data = read_data("./input_data/burma14.txt")

args = get_args()

for seed in args.seeds:
    for size_adj in args.size_adjs:
        try:
            run(RunParams(seed, args.steps, size_adj, data, args.freq, args.tag))
        except:
            print("FAILURE DURING RUN")
            traceback.print_exc()
