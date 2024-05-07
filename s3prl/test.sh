#!/bin/bash -l
#SBATCH --time=01:00:00
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

CUDA_VISIBLE_DEVICES=0

RESULT_DIR=/data/lmorove1/hwang258/data/commonvoice/s3prl/s3prl/result
META_DIR=/data/lmorove1/hwang258/data/commonvoice/metadata

python3 run_downstream.py -m test \
    -t ${META_DIR}/alldata_TODO.csv \
    -x ${META_DIR}/alldata.csv \
    -u hubert -e ${RESULT_DIR}/downstream/DreamVoiceBrightHubertSpecAug/dev-best.ckpt

python3 run_downstream.py -m test \
    -t ${META_DIR}/alldata.csv \
    -x ${META_DIR}/alldata.csv \
    -u hubert -e ${RESULT_DIR}/downstream/DreamVoiceSmoothHubertSpecAug/dev-best.ckpt

python3 run_downstream.py -m test \
    -t ${META_DIR}/alldata.csv \
    -x ${META_DIR}/alldata.csv \
    -u hubert -e ${RESULT_DIR}/downstream/SpeechEmotionHubert/dev-best.ckpt

python3 run_downstream.py -m test \
    -t ${META_DIR}/alldata.csv \
    -x ${META_DIR}/alldata.csv \
    -u hubert -e ${RESULT_DIR}/downstream/CommonVoiceGenderHubert/dev-best.ckpt
