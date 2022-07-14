#!/bin/sh

#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhang.tian@duke.edu
module load python3
srun python3 Experiment_MDP.py