#!/bin/sh
#SBATCH --job-name=FRUIT
#SBATCH --nodes=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --ntasks=1                    
#SBATCH --cpus-per-task=48            
#SBATCH --mem=100g
#SBATCH --time=4-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=FRUIT%J.out
#SBATCH --error=FRUIT%J.err
python3 fruit_tree.py