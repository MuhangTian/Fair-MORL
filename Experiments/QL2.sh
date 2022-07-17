#!/bin/sh
#SBATCH --job-name=QL_PEN
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=48            
#SBATCH --mem=50g
#SBATCH --time=4-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=QL_PEN.out.%J
#SBATCH --error=QL_PEN.err.%J
python3 ql_taxi_pen.py