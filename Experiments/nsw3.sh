#!/bin/sh
#SBATCH --job-name=CONT_NSW
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=48            
#SBATCH --mem=50g
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=CONT_NSW.out.%J
#SBATCH --error=CONT_NSW.err.%J
python3 nsw_ql_taxi.py -f 500000000 -ep 1 -n 2