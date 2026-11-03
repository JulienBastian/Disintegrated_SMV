#!/usr/bin/env bash
#SBATCH --time=2-00:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 18
#SBATCH -p LONG3,LONG2,LONG,GPU-UJM

# From the repository root, with the environment activated:
#   sbatch rebuttal/cbound/cbound_slurm.sh
export MAX_JOBS="${MAX_JOBS:-18}"
export PYTHONWARNINGS=ignore

./rebuttal/cbound/cbound.sh
