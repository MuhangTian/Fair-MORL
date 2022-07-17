#!/bin/sh
#SBATCH --job-name=BIG_NSW
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=48            
#SBATCH --mem=50g
#SBATCH --time=4-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=BIG_NSW.out.%J
#SBATCH --error=BIG_NSW.err.%J
python3 nsw_ql_taxi2.py -f 500000000 -ep 1 -n 1