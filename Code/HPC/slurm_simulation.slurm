#!/bin/bash

#SBATCH --job-name=TempRanges
#SBATCH --account=biotheo.biotheo         # Your account
#SBATCH --partition=smp
#SBATCH --time=0:30:00
#SBATCH --ntasks=1

# run 1000 tasks, but only run 20 at a time
#SBATCH --array=1-1%1
#SBATCH --output=Outputs/run_1/result_%A_%a.out    # gives result_<jobID>_<taskID>.out
 
echo "SLURM_JOBID:         $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_JOB_ID:  $SLURM_ARRAY_JOB_ID"

# load julia
module purge
module load julia/1.7.2

# run julia script
srun julia ./Code/HPC/slurm_script.jl $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID "/albedo/work/user/thcleg001/Projects/TempRanges/Data/run_1"