#!/bin/sh
#SBATCH --job-name=TAXI1
#SBATCH --nodes=1
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=32            # Number of CPU cores per task
#SBATCH --mem-per-cpu=2g
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
module load python3
python3 Experiment_MDP.py