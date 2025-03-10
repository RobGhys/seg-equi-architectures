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
#SBATCH --account=lysmed


# ------------------------- work -------------------------
# Setting the number of worker for data loading
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/cudnn/lib64:$LD_LIBRARY_PATH

export PYTHONPATH=$PYTHONPATH:/gpfs/scratch/acad/lysmed/seg-equi-architectures/src/U-Net

# Check if wandb_api_key is provided as a command line argument
if [ "$#" -ne 1 ]; then
    echo "Error: You must provide the wandb_api_key as an argument."
    echo "Usage: sbatch training.sh <wandb_api_key>"
    exit 1
fi

# Assign the first command line argument to wandb_api_key
wandb_api_key=$1


echo "CUDA environment:"
which nvcc
nvidia-smi
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
ldconfig -p | grep libnvrtc

echo "Starting Task #: $SLURM_ARRAY_TASK_ID"

#resume_path="/gpfs/scratch/acad/lysmed/seg-equi-architectures/outputs/coco/UNet_e2cnn/fold_${SLURM_ARRAY_TASK_ID}/checkpoint_epoch_159.pth"

# Check if the resume file exists
#if [ ! -f "$resume_path" ]; then
#    echo "Error: Resume file not found at $resume_path"
#    exit 1
#fi

python src/Benchmarks/training/main.py \
kvasir \
UNet_e2cnn \
$SLURM_ARRAY_TASK_ID \
--save_logs \
--location_lucia \
--wandb_api_key $wandb_api_key \
--save_model \
--freq-save-model 50
#--use_amp
#--resume "$resume_path" \
#--start-epoch 160 \

echo "Finished Task #: $SLURM_ARRAY_TASK_ID"

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo "Python script failed with exit code $exit_code"
    echo "Checking CUDA libraries..."
    find /usr -name "libnvrtc.so*" 2>/dev/null
    find /usr -name "libcudnn_ops_infer.so.8*" 2>/dev/null
fi

echo "Finished Task #: $SLURM_ARRAY_TASK_ID"

echo "Exiting the program."