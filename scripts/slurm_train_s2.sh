#!/bin/bash
#SBATCH --job-name=hora_s2
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
python -u train.py task=AllegroHandHora headless=True seed=0 \
  task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
  train.algo=ProprioAdapt \
  task.env.object.type=cylinder_default \
  train.ppo.priv_info=True train.ppo.proprio_adapt=True \
  train.ppo.output_name=AllegroHandHora/s2_${SLURM_JOB_ID} \
  checkpoint=outputs/AllegroHandHora/"$1"/stage1_nn/best.pth "${@:2}"
