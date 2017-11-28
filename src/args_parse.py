import argparse


def get_args():
    parser = argparse.ArgumentParser(description="result analyser for MpiTsp project")
    parser.add_argument('--steps', nargs="?", default=2000, type=int,
                        help='Number of steps to take.')
    parser.add_argument('--freq', nargs='?', default=10, type=int,
                        help='Frequency of taking snapshots.')
    parser.add_argument('--seeds', nargs='*', default=[1], type=int,
                        help='Seed for random. Defines whole run.')
    parser.add_argument('--size-adjs', nargs='*', default=[0], type=float,
                        help='specifies value of size adjustment')
    parser.add_argument('--tag', nargs='?', default="", type=str,
                        help='tag added to name')
    args = parser.parse_args()
    return args
