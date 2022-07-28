#!/bin/sh
#SBATCH --job-name=NSW24
#SBATCH --nodes=1
#SBATCH --mail-type=end
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=24            
#SBATCH --mem=25g
#SBATCH --time=4-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=%J.out
#SBATCH --error=%J.err
python3 nsw_ql_taxi_pen.py -n 1 -locs [[0,0],[0,5],[3,2]] -dests [[0,4],[5,0],[3,3]] -gs 10