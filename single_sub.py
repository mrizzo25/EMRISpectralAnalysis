import numpy as np
import argparse
import os

import sys

jobid = sys.getenv('SLURM_ARRAY_TASK_ID')




parser = argparse.ArgumentParser()
parser.add_argument("--fname", type=str, help='filename of output')
parser.add_argument("--ini", type=str, help='file with default parameters')
parser.add_argument("--grid-param", type=str, action="append", help="param to grid over")
parser.add_argument("--n-grid", nargs="+", type=int, help="number of points for each param on grid")

parser.add_argument("--log-scale-param", type=str, action="append", help="use log spacing for this params on grid")

parser.add_argument("--p-isco-fraction", default=None, help="fix slr to fractional value of isco slr")
parser.add_argument("--p-mb-fraction", default=None, help="fix value of marginally bound radius")
parser.add_argument("--fix-sma", default=None, help="fix semimajor axis value")

parser.add_argument("--overwrite", action='store_true', help="if output data file already exists, overwrite it")
parser.add_argument("--verbose", action='store_true', help="print extra debugging output")

parser.add_argument("--slurm-opts", type=str, help="string containing slurm options. format as 'option1=val option2=val ...'")
parser.add_argument("--slurm-module", type=str, action="append", help="one or more modules to load in slurm scripts")
parser.add_argument("--slurm-env", type=str, default=None, help="environment to source for slurm scripts")

parsed, unknown = parser.parse_known_args()

#accept additional params as emri params
for arg in unknown:

    #if passed a parameter grid code recognizes, make a valid argument
    if arg.startswith(("--", "-")) and (arg.replace('--', '') in param_type_dict.keys() or \
            arg.replace('-', '') in param_type_dict.keys()):

        if arg.startswith("--"):
            key = arg.replace('--', '')
        else:
            key = arg.replace('-', '')

        parser.add_argument(arg, type=param_type_dict[key])

args = parser.parse_args()


