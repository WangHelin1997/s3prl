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

python3 run_downstream.py -m train -n CommonVoiceAgeHubert -u hubert -d age -s hidden_states

# python3 run_downstream.py -m train -n SpeechEmotionHubert -u hubert -d emotion -s hidden_states -l 4 -f

python3 run_downstream.py -m train -n CommonVoiceGenderHubert -u hubert -d gender -s hidden_states -l 4 -f

# python3 run_downstream.py -m train -n DreamVoiceBrightHubert -u hubert -d bright -s hidden_states -l 4 -f

# python3 run_downstream.py -m train -n DreamVoiceSmoothHubert -u hubert -d smooth -s hidden_states -l 4 -f

# python3 run_downstream.py -m train -n CommonVoiceAgeHubertSpecAug -u hubert -d age -s hidden_states -l 4 -f

# python3 run_downstream.py -m train -n CommonVoiceAgeHubertAll -u hubert -d age -s hidden_states -f

# python3 run_downstream.py -m train -n SpeechEmotionHubertSpecAug -u hubert -d emotion -s hidden_states -l 4 -f

# python3 run_downstream.py -m train -n DreamVoiceBrightHubertSpecAug -u hubert -d bright -s hidden_states -l 4 -f

# python3 run_downstream.py -m train -n DreamVoiceSmoothHubertSpecAug -u hubert -d smooth -s hidden_states -l 4 -f
