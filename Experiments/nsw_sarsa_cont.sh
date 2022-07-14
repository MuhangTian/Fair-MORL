#!/bin/sh
#SBATCH --job-name=CONT_SARSA
#SBATCH --nodes=1
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=4            
#SBATCH --mem-per-cpu=6g
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=CONT_SARSA.out.%J
#SBATCH --error=CONT_SARSA.err.%J
module load python3
python3 nsw_sarsa_taxi.py -f 500000000 -ep 1 -n 2