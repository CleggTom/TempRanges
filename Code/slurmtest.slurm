#!/bin/bash

#SBATCH --account=biotheo.biotheo         # Your account
#SBATCH --partition=smp
#SBATCH --time=0:10:00
#SBATCH --ntasks=1
 
# run 100 tasks, but only run 10 at a time
#SBATCH --array=1-2%2
#SBATCH --output=Outputs/result_%A_%a.out    # gives result_<jobID>_<taskID>.out
 
echo "SLURM_JOBID:         $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID:  $SLURM_ARRAY_JOB_ID"

# load julia
module purge
module load julia/1.7.2

# run julia script
srun julia ./Code/slurm_script.jl $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID