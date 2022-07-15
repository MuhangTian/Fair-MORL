#!/bin/sh
#SBATCH --job-name=BIG_NSW_2
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=48            
#SBATCH --mem=50g
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=BIG_NSW_2.out.%J
#SBATCH --error=BIG_NSW_2.err.%J
python3 nsw_ql_taxi2.py -f 500000000 -ep 1 -e 0.1 -n 2