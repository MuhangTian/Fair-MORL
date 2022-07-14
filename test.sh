#!/bin/sh
#SBATCH --job-name=TEST
#SBATCH --nodes=1
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=32            # Number of CPU cores per task
#SBATCH --mem-per-cpu=2g
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
module load python3
python3 scratch.py