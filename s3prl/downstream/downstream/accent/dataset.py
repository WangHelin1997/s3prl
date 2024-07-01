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
import pandas as pd

SAMPLE_RATE = 16000


class SpeechAccentDataset(Dataset):
    def __init__(self, meta_path, split, segment_size=None, base_dir=None):
        df = pd.read_csv(meta_path)
        filtered_df = df[df['split'] == split]
        filtered_df.reset_index(drop=True, inplace=True)
        self.data = filtered_df
        self.class_num = 23
        self.segment_size = segment_size
        self.base_dir = base_dir
        self.label_dict = {
            "Dutch":0,
            "German":1,
            "Czech":2,
            "Polish":3,
            "French":4,
            "Hungarian":5,
            "Finnish":6,
            "Romanian":7,
            "Slovak":8,
            "Spanish":9,
            "Italian":10,
            "Estonian":11,
            "Lithuanian":12,
            "Croatian":13,
            "Slovene":14,
            "English":15,
            "Scottish":16,
            "Irish":17,
            "NorthernIrish":18,
            "Indian":19,
            "Vietnamese":20,
            "Canadian":21,
            "American":22
            }
    def _load_wav(self, path):
        tag = False
        while tag == False:
            try:
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
                tag = True
            except:
                path = self.data['audio'][random.randint(0, 50000)]

    def __getitem__(self, idx):
        labelid = self.data['accent'][idx]
        # tags = ["Dutch","German","Czech","Polish","French","Hungarian","Finnish","Romanian","Slovak","Spanish","Italian","Estonian","Lithuanian","Croatian","Slovene","English","Scottish","Irish","NorthernIrish","Indian","Vietnamese","Canadian","American"]
        label = self.label_dict[labelid]
        wav = self._load_wav(self.data['audio'][idx])

        return wav.numpy(), label, self.data['audio_id'][idx]

    def __len__(self):
        return len(self.data)

def collate_fn(samples):
    return zip(*samples)
