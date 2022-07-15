#!/bin/sh
#SBATCH --job-name=NSW_DIFF
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=48            
#SBATCH --mem=50g
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=NSW_DIFF.out.%J
#SBATCH --error=NSW_DIFF.err.%J
python3 nsw_ql_taxi.py