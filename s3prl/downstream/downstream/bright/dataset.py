# -*- coding: utf-8 -*- #
"""
    FileName     [ dataset.py ]
    Synopsis     [ the emotion classifier dataset ]
    Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""

import json
from pathlib import Path
from os.path import join as path_join

import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
import csv
import random
import torch

SAMPLE_RATE = 16000


class DreamVoiceBrightDataSet(Dataset):
    def __init__(self, meta_path, segment_size=None):
        with open(meta_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            self.data = list(reader)
            
        self.class_num = 2
        self.segment_size = segment_size
        
    def _load_wav(self, path):
        wav, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            self.resampler = Resample(sr, SAMPLE_RATE)
            wav = self.resampler(wav)
        wav = wav.squeeze(0)
        if wav.shape[0] >= self.segment_size:
            max_audio_start = wav.shape[0] - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            wav = wav[audio_start:audio_start + self.segment_size]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.segment_size - wav.shape[0]), 'constant')
            
        return wav

    def __getitem__(self, idx):
        labelid = self.data[idx][1]
        if labelid =='Bright':
            label = 0
        elif labelid =='Dark':
            label = 1

        wav = self._load_wav(self.data[idx][0])
        return wav.numpy(), label, self.data[idx][0]

    def __len__(self):
        return len(self.data)

def collate_fn(samples):
    return zip(*samples)
