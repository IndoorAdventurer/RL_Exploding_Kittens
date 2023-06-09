#!/bin/bash

#SBATCH --job-name=exploding_kittens_drl
#SBATCH --mail-type=ALL
#SBATCH --mail-user=v.tonkes@student.rug.nl
#SBATCH --time=7:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100.20gb:1

module purge
module load Python/3.10.8-GCCcore-12.2.0

source $HOME/.envs/ek_drl_env/bin/activate

python $1

deactivate
