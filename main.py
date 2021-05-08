#!/usr/bin/env python3

import argparse

from regression_lasso.reg_sbm import run_reg_sbm_2blocks, run_reg_sbm_5blocks
from regression_lasso.reg_3d_road.reg_merge_3d_road import run_reg_merge_3d_road
from regression_lasso.reg_complete import run_reg_complete
from deep_learning_lasso.main import deep_learning_run


def parse_arguments():
    """Read arguments from a command line."""
    parser = argparse.ArgumentParser(description='Arguments get parsed via --commands')
    parser.add_argument('-n', '--name', type=str, default='sbm_2',
                        help='Experiment name: sbm_2, sbm_5, 3d_road, complete, deep_learning')

    parser.add_argument('-l', '--lambda_lasso', type=float, default=0.001,
                        help='lambda parameter of the algorithm which is a float')

    parser.add_argument('-i', '--iters', type=int, default=1000,
                        help='number of iterations for the algorithm which is an int')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_arguments()
    lambda_lasso = args.lambda_lasso
    K = args.iters
    if args.name == 'sbm_2':
        print(run_reg_sbm_2blocks())
    elif args.name == 'sbm_5':
        print(run_reg_sbm_5blocks())
    elif args.name == '3d_road':
        print(run_reg_merge_3d_road())
    elif args.name == 'complete':
        print (run_reg_complete())
    elif args.name == 'deep_learning':
        print(deep_learning_run())
    else:
        print("invalid experiment name")

