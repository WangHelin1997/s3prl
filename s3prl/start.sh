#!/bin/bash -l
#SBATCH --time=12:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu
#SBATCH --account=lmorove1_gpu 
#SBATCH --job-name=”s3prl”

source activate wav2vec
ml purge
ml

nvidia-smi

python3 run_downstream.py -m train -n CommonVoiceAgeHubertSpecAug -u hubert -d age -s hidden_states

python3 run_downstream.py -m train -n CommonVoiceGenderHubertSpecAug -u hubert -d gender -s hidden_states
