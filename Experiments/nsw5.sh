#!/bin/sh
#SBATCH --job-name=NSW_PEN3
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=48            
#SBATCH --mem=50g
#SBATCH --time=4-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=NSW_PEN3.out.%J
#SBATCH --error=NSW_PEN3.err.%J
python3 nsw_ql_taxi_pen.py -g 0.85 -n 3