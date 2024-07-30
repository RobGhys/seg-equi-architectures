#!/bin/bash
#
#SBATCH --job-name=training_seg
#SBATCH --array=0-4
#SBATCH --time=06:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:1"
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4096
#SBATCH --partition=gpu
#
#SBATCH --mail-user=robin.ghyselinck@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=bcnn


# ------------------------- work -------------------------
echo "hi"

echo "Exiting the program."