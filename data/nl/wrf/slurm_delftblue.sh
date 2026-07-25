#!/bin/bash
# Job file for DelftBlue. Missing ressources are specfied in the sbatch call.
#SBATCH --partition=compute
#SBATCH --account=research-ceg-grs
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --mail-type=END,FAIL

# Set variables
STAGE=$1
SIM_DIR=$2

# If job is run as slurm array, read SIM_DIR from file.
# Otherwise, use SIM_DIR from command line argument.
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  SIM_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" .array_sim_dirs)
  echo "Running array job with ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}, SIM_DIR=$SIM_DIR"
fi

if [ -z "$STAGE" ] || [ -z "$SIM_DIR" ]; then
  echo "Usage: sbatch slurm_delftblue.job.sh <stage> <sim_dir>"
  exit 1
fi

# Load modules
source /home/mpierzyna/wrf/wrf_wps_env.sh
module load slurm

# Make tmp WPS dir on node-local SSD
mkdir -p /tmp/wrf-massive

# Run (unbuffered)
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK \
OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK \
MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK \
VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK \
NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK \
PYTHONUNBUFFERED=1 uv run cli.py run --stages=$STAGE $SIM_DIR
