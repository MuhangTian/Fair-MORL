#!/bin/sh
#SBATCH --job-name=BIG_PEN2
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=48            
#SBATCH --mem=50g
#SBATCH --time=4-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=BIG_PEN2.out.%J
#SBATCH --error=BIG_PEN2.err.%J
python3 nsw_ql_taxi_pen2.py -g 0.9 -n 2