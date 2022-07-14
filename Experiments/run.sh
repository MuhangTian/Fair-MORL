#!/bin/sh
#SBATCH --job-name=AGENT-BONT
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=1gb                     # Job memory request
#SBATCH --time=04:00:00               # Time limit hrs:min:sec
mt361
module load python3
srun python3 Experiment_MDP.py