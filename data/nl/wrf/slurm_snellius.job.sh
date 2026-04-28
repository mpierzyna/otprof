#!/bin/bash
# Job file for Snellius. Missing ressources are specfied in the sbatch call.
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --mail-user=m.pierzyna@tudelft.nl
#SBATCH --mail-type=END,FAIL

# Set variables
CONDA_ENV=/home/mpierzyna/.conda/envs/wrf_p312
STAGE=$1
SIM_DIR=$2

# If job is run as slurm array, read SIM_DIR from file.
# Otherwise, use SIM_DIR from command line argument.
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  SIM_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" .array_sim_dirs)
  echo "Running array job with ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}, SIM_DIR=$SIM_DIR"
fi

if [ -z "$STAGE" ] || [ -z "$SIM_DIR" ]; then
  echo "Usage: sbatch slurm_snellius.job.sh <stage> <sim_dir>"
  exit 1
fi

# Load conda environment
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

# Load modules
module load 2024
module load foss/2024a

# Run (unbuffered)
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \
OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK \
MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK \
VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK \
NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK \
python -u cli.py run --stages=$STAGE $SIM_DIR
