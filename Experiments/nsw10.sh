#!/bin/sh
#SBATCH --job-name=NSW10
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=24            
#SBATCH --mem=25g
#SBATCH --time=4-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=%J.out
#SBATCH --error=%J.err
python3 nsw_ql_taxi_pen_v2.py -n 2 -g 0.9 -locs [[0,0], [0,5], [3,2], [9,0], [8,9], [5,5], [7,4], [1,7], [6,2]] -dests [[0,4], [5,0], [3,3], [0,9], [4,7], [5,9], [8,4], [6,8], [8,6]]