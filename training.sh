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

# Setting the number of worker for data loading
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

export PYTHONPATH=$PYTHONPATH:/gpfs/projects/acad/bcnn/seg-equi-architectures/src/U-Net

echo "Starting Task #: $SLURM_ARRAY_TASK_ID"
python src/Benchmarks/training/main.py kvasir UNet_vanilla $SLURM_ARRAY_TASK_ID --save_logs --save_images
echo "Finished Task #: $SLURM_ARRAY_TASK_ID"

echo "Exiting the program."