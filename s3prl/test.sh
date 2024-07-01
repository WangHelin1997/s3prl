#!/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --qos=qos_gpu
#SBATCH --account=lmorove1_gpu 
#SBATCH --job-name=”s3prl”

# source activate wav2vec
# ml purge
# ml

# nvidia-smi

# CUDA_VISIBLE_DEVICES=0

RESULT_DIR=/data/lmorove1/hwang258/s3prl/s3prl/result
SAVE_DIR=/data/lmorove1/hwang258/dataspeech/cache/
AUDIO_DIR=/data/lmorove1/hwang258/dataspeech/cache/audios/

CUDA_VISIBLE_DEVICES=0 python3 run_downstream.py -m inference \
    -t  ${AUDIO_DIR} \
    -x ${SAVE_DIR}/age.csv \
    --tag age \
    -u hubert -e ${RESULT_DIR}/downstream/CommonVoiceAgeHubertSpecAug/dev-best.ckpt

CUDA_VISIBLE_DEVICES=0 python3 run_downstream.py -m inference \
    -t  ${AUDIO_DIR} \
    -x ${SAVE_DIR}/gender.csv \
    --tag gender \
    -u hubert -e ${RESULT_DIR}/downstream/CommonVoiceGenderHubertSpecAug/dev-best.ckpt

CUDA_VISIBLE_DEVICES=0 python3 run_downstream.py -m inference \
    -t ${AUDIO_DIR} \
    -x ${SAVE_DIR}/brightness.csv \
    --tag brightness \
    -u hubert -e ${RESULT_DIR}/downstream/DreamVoiceBrightHubertSpecAug/dev-best.ckpt

CUDA_VISIBLE_DEVICES=0 python3 run_downstream.py -m inference \
    -t ${AUDIO_DIR} \
    -x ${SAVE_DIR}/smoothness.csv \
    --tag smoothness \
    -u hubert -e ${RESULT_DIR}/downstream/DreamVoiceSmoothHubertSpecAug/dev-best.ckpt

CUDA_VISIBLE_DEVICES=0 python3 run_downstream.py -m inference \
    -t ${AUDIO_DIR} \
    -x ${SAVE_DIR}/accent.csv \
    --tag accent \
    -u hubert -e ${RESULT_DIR}/downstream/SpeechAccentHubertSpecAug/dev-best.ckpt

CUDA_VISIBLE_DEVICES=0 python3 run_downstream.py -m inference \
    -t ${AUDIO_DIR} \
    -x ${SAVE_DIR}/emotion.csv \
    --tag emotion \
    -u hubert -e ${RESULT_DIR}/downstream/SpeechEmotionHubert/dev-best.ckpt