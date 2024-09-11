#!/bin/bash
#
#SBATCH --job-name=training_seg
#SBATCH --array=0-4
#SBATCH --time=48:00:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --gres="gpu:1"
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=7650
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

# Check if wandb_api_key is provided as a command line argument
if [ "$#" -ne 1 ]; then
    echo "Error: You must provide the wandb_api_key as an argument."
    echo "Usage: sbatch training.sh <wandb_api_key>"
    exit 1
fi

# Assign the first command line argument to wandb_api_key
wandb_api_key=$1

echo "Starting Task #: $SLURM_ARRAY_TASK_ID"

resume_path="/gpfs/projects/acad/bcnn/seg-equi-architectures/outputs/coco/UNet_e2cnn/fold_${SLURM_ARRAY_TASK_ID}/checkpoint_epoch_159.pth"

# Check if the resume file exists
if [ ! -f "$resume_path" ]; then
    echo "Error: Resume file not found at $resume_path"
    exit 1
fi

python src/Benchmarks/training/main.py \
coco \
UNet_e2cnn \
$SLURM_ARRAY_TASK_ID \
--save_logs \
--location_lucia \
--wandb_api_key $wandb_api_key \
--save_model \
--freq-save-model 100 \
--resume "$resume_path" \
--start-epoch 160 \
--use_amp

echo "Finished Task #: $SLURM_ARRAY_TASK_ID"

echo "Exiting the program."

