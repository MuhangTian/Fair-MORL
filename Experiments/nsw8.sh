#!/bin/sh
#SBATCH --job-name=NSW8
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=48            
#SBATCH --mem=50g
#SBATCH --time=4-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=NSW8.out.%J
#SBATCH --error=NSW8.err.%J
python3 nsw_ql_taxi_pen_v2_2.py -n 4 -g 0.8