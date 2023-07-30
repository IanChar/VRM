"""
Launch several training jobs.

Author: Ian Char
Date: February 16, 2023
"""
import argparse
from dataclasses import dataclass
import datetime
from typing import Any, Dict
import os
import subprocess
import time

import numpy as np

# Parse Arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--script', type=str,
                    default='run_experiment.py')
parser.add_argument('--env', type=str)
parser.add_argument('--num_seeds', type=int,
                    default=5)
parser.add_argument('--num_gpus', type=int,
                    default=1000)
parser.add_argument('--jobs_per_gpu', type=int,
                    default=1000)
args = parser.parse_args()

# Initialize global variables.
LOG_PATH = os.path.join('logs', f'{datetime.datetime.now()}.txt')
GPU_COUNTS = [0 for _ in range(args.num_gpus)]
MAX_RUNNING = sum([args.jobs_per_gpu - gc for gc in GPU_COUNTS])
RUNNING = []
seed_offset = 0
ARG_DICT = {
    'seed': [i + seed_offset for i in range(args.num_seeds)],
    'env': args.env.split(','),
}


@dataclass
class Job:
    proc: subprocess.Popen
    gpu: int
    job_args: Dict[str, Any]


def prune_completed_job():
    for jidx, job in enumerate(RUNNING):
        if job.proc.poll() is not None:
            GPU_COUNTS[job.gpu] -= 1
            with open(LOG_PATH, 'a') as f:
                f.write(f'{datetime.datetime.now()}\t Finished \t {job.job_args}\n')
            RUNNING.pop(jidx)
            return True
    return False


def add_job(job_args):
    if len(RUNNING) >= MAX_RUNNING:
        while not prune_completed_job():
            time.sleep(30)
    with open(LOG_PATH, 'a') as f:
        f.write(f'{datetime.datetime.now()}\t Starting \t {job_args}\n')
    # Find open gpu device.
    gpu = 0
    while gpu < len(GPU_COUNTS) - 1 and GPU_COUNTS[gpu] >= args.jobs_per_gpu:
        gpu += 1
    GPU_COUNTS[gpu] += 1
    cmd = (f'python {args.script} run --env={job_args["env"]} '
           f'--steps=1000000 --seed={job_args["seed"]}')
    proc = subprocess.Popen(cmd, shell=True)
    RUNNING.append(Job(proc, gpu, job_args))


# Run it!
if not os.path.exists('logs'):
    os.makedirs('logs')
with open(LOG_PATH, 'w') as f:
    f.write('Timestamp \t Status \t Args\n')
arg_keys = list(ARG_DICT.keys())
num_each_args = np.array([len(ARG_DICT[k]) for k in arg_keys])
arg_idxs = np.array([0 for _ in range(len(ARG_DICT))])
while True:
    add_job({k: ARG_DICT[k][arg_idxs[kidx]] for kidx, k, in enumerate(arg_keys)})
    arg_idxs[0] += 1
    for ii in range(len(arg_idxs) - 1):
        if arg_idxs[ii] >= num_each_args[ii]:
            arg_idxs[ii] = 0
            arg_idxs[ii + 1] += 1
    if np.any(arg_idxs >= num_each_args):
        break
while len(RUNNING) > 0:
    prune_completed_job()
    time.sleep(30)
with open(LOG_PATH, 'w') as f:
    f.write('Done!')
