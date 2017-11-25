import argparse


def get_args():
    parser = argparse.ArgumentParser(description="result analyser for MpiTsp project")
    parser.add_argument('--steps', nargs="?", default=2000, type=int,
                        help='argument for specyfing included number of threads')
    parser.add_argument('--freq', nargs='?', default=10, type=int,
                        help='argument for specyfing expected numbers of population')
    parser.add_argument('--seed', nargs='?', default=1, type=int,
                        help='argument for specyfing expected numbers of population')
    args = parser.parse_args()
    return args
