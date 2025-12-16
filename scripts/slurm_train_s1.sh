#!/bin/bash
#SBATCH --job-name=hora_s1
#SBATCH --output=../logs/%j_%x.txt
#SBATCH --error=../logs/%j_%x.txt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH -C rtx&vram24gb

eval "$(conda shell.bash hook)"
conda activate hora

cd ..
python train.py task=AllegroHandHora headless=True seed=0 \
  task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
  train.algo=PPO \
  task.env.object.type=cylinder_default \
  train.ppo.priv_info=True train.ppo.proprio_adapt=False \
  train.ppo.output_name=AllegroHandHora/s1_${SLURM_JOB_ID} \
  num_envs=65536
