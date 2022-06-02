# -*- coding: utf-8 -*-

import numpy as np
from joblib import Parallel, delayed
import librosa
import soundfile as sf
import os
from glob import glob
from tqdm import tqdm
import random
import json
import resampy
import pyworld as pw

from spectrogram import logmelspectrogram
from utils import get_formant

def extract_logmel(wav_path, sr=16000):
    # wav, fs = librosa.load(wav_path, sr=sr)
    wav, fs = sf.read(wav_path)
    wav, _ = librosa.effects.trim(wav, top_db=60)
    if fs != sr:
        wav = resampy.resample(wav, fs, sr, axis=0)
        fs = sr
    # duration = len(wav)/fs
    assert fs == 16000
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
    mel = logmelspectrogram(
                x=wav,
                fs=fs,
                n_mels=80,
                n_fft=1024,
                n_shift=160,
                win_length=400,
                window='hann',
                fmin=80,
                fmax=7600,
            )
    
    tlen = mel.shape[0]
    frame_period = 160/fs*1000
    f0, timeaxis = pw.dio(wav.astype('float64'), fs, frame_period=frame_period)
    f0 = pw.stonemask(wav.astype('float64'), f0, timeaxis, fs)
    f0 = f0[:tlen].reshape(-1).astype('float32')
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices]) # for f0(Hz), lf0 > 0 when f0 != 0
    
    wav_name = f"{wav_path.split('/')[-2]}_{os.path.basename(wav_path).split('.')[0]}"

    # get formant
    F1 = get_formant(wav_path, f0, formant_num=1)
    F2 = get_formant(wav_path, f0, formant_num=2) 
    return wav_name, mel, f0, mel.shape[0], F1, F2


def normalize_logmel(wav_name, mel, mean, std):
    mel = (mel - mean) / (std + 1e-8)
    return wav_name, mel


def save_one_file(save_path, arr):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, arr)

# def get_wavs_names(spks, data_root)
data_root = '../MonoData/wavs'
save_root = 'data_F002'
os.makedirs(save_root, exist_ok=True)

# extract log-mel
print('extract log-mel...')
all_wavs = glob(f'{data_root}/F002/*.wav')
results = Parallel(n_jobs=-1)(delayed(extract_logmel)(wav_path) for wav_path in tqdm(all_wavs))
mels = []
wn2mel = {}
for r in results:
    wav_name, mel, lf0, mel_len, F1, F2 = r
    mels.append(mel)
    wn2mel[wav_name] = [mel, lf0, mel_len, F1, F2]

# normalize log-mel
print('normalize log-mel...')
mels = np.concatenate(mels, 0)
mean = np.mean(mels, 0)
std = np.std(mels, 0)
mel_stats = np.concatenate([mean.reshape(1,-1), std.reshape(1,-1)], 0)
np.save(f'{save_root}/mel_stats.npy', mel_stats)

results = Parallel(n_jobs=-1)(delayed(normalize_logmel)(wav_name, wn2mel[wav_name][0], mean, std) for wav_name in tqdm(wn2mel.keys()))
wn2mel_new = {}
for r in results:
    wav_name, mel = r
    lf0 = wn2mel[wav_name][1]
    mel_len = wn2mel[wav_name][2]
    F1 = wn2mel[wav_name][3]
    F2 = wn2mel[wav_name][4]
    wn2mel_new[wav_name] = [mel, lf0, mel_len, F1, F2]

train_wavs_names = []
print('all_wavs:', len(all_wavs))
for wav_path in all_wavs:
    wav_name = f"{wav_path.split('/')[-2]}_{os.path.basename(wav_path).split('.')[0]}"
    train_wavs_names.append(wav_name)

def save_logmel(save_root, wav_name, melinfo, mode):
    mel, lf0, mel_len, F1, F2 = melinfo
    mel_save_path = f'{save_root}/{mode}/mels/{wav_name}.npy'
    lf0_save_path = f'{save_root}/{mode}/f0/{wav_name}.npy'
    F1_save_path = f'{save_root}/{mode}/F1/{wav_name}.npy'
    F2_save_path = f'{save_root}/{mode}/F2/{wav_name}.npy'
    save_one_file(mel_save_path, mel)
    save_one_file(lf0_save_path, lf0)
    save_one_file(F1_save_path, F1)
    save_one_file(F2_save_path, F2)

# save log-mel
print('save log-mel...')
Parallel(n_jobs=-1)(delayed(save_logmel)(save_root, wav_name, wn2mel_new[wav_name], 'F002') for wav_name in tqdm(train_wavs_names))
