#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=story_generation
#SBATCH --account=si630w24_class
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16000m
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=story_generation.log

# The application(s) to execute along with its input arguments and options:

/bin/hostname

# Find the latest checkpoint file by modification time
checkpoint=$(ls -t ./results/checkpoint-* | head -1)  # '-t' sorts by modification time, newest first

if [ -n "$checkpoint" ]; then
    checkpoint=${checkpoint%:}
    echo "Checkpoint is '$checkpoint'"
    python mamba_story.py --resume_from_checkpoint="$checkpoint"
    nvidia-smi
else
    python mamba_story.py
    nvidia-smi
fi

