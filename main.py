import json
import munch
import os
import argparse

import torch
import numpy as np

from impulse_control_solver import ImpulseControlSolver


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help="The path to load json file")
    parser.add_argument('--run_name', type=str, help='The name of numerical experiments')
    return parser.parse_args()


def load_config(args):
    with open(args.config_path) as json_data_file:
        config = json.load(json_data_file)
    cfg = munch.munchify(config)
    cfg.base_dir = os.path.dirname(os.path.abspath(args.config_path))
    return cfg


def main():
    print(torch.__version__)
    args = parse_arguments()
    config = load_config(args)
    #prepare_directories(args)

    impulse_solver = ImpulseControlSolver(config, run_name=args.run_name)
    impulse_solver.train()


if __name__ == '__main__':
    main()
