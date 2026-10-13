#!/usr/bin/env bash

# Use N_CPUS from SLURM if available
if [ -n "$SLURM_NTASKS" ]; then
  N_CPUS=$SLURM_NTASKS
fi

# If RUNNER is not specified, default to mpirun.
# If your mpi is compiled with slurm support, you can set RUNNER="srun" in the job script.
if [ -z "$RUNNER" ]; then
  RUNNER="mpirun -np $N_CPUS"
fi

if [ -z "$N_CPUS" ]; then
  # Serial run, if N_CPUS not set from SLURM (above) or externally
  echo "SERIAL MODE!"
  ./real.exe || exit 1
  ./wrf.exe  || exit 2
else
  # Parallel run
  echo "PARALLEL MODE with $N_CPUS CPUs!"
  $RUNNER ./real.exe || exit 1
  $RUNNER ./wrf.exe  || exit 2
fi
