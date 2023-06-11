#!/bin/bash

#SBATCH --job-name=drl😺
#SBATCH --mail-type=ALL
#SBATCH --mail-user=v.tonkes@student.rug.nl
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100.20gb:1
#SBATCH --cpus-per-task=5

module purge
module load Python/3.10.8-GCCcore-12.2.0

source $HOME/.envs/ek_drl_env/bin/activate

python $1

deactivate
