import numpy as np
import torch
from torch.utils.data import Dataset
import json
import random
from pathlib import Path
import os

from utils import log


class CPCDataset_sameSeq(Dataset):
    def __init__(self, root, n_sample_frames, mode):
        self.root = Path(root)
        self.n_sample_frames = n_sample_frames

        self.speakers = sorted(os.listdir(root/f'{mode}/mels'))
        
        with open(self.root / f"{mode}.json") as file:
            metadata = json.load(file)
        self.metadata = []
        for mel_len, mel_out_path, lf0_out_path in metadata:
            # if mel_len > n_sample_frames: # only select wavs having frames>=140
            mel_out_path = Path(mel_out_path)
            lf0_out_path = Path(lf0_out_path)
            speaker = mel_out_path.parent.stem
            self.metadata.append([speaker, mel_out_path, lf0_out_path])
        print('n_sample_frames:', n_sample_frames, 'metadata:', len(self.metadata))
        random.shuffle(self.metadata)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        speaker, mel_path, lf0_path = self.metadata[index]

        mel_path = self.root.parent / mel_path
        lf0_path = self.root.parent / lf0_path
        mel = np.load(mel_path).T
        lf0 = np.load(lf0_path)
        melt = mel
        lf0t = lf0
        while mel.shape[-1] < self.n_sample_frames:
            mel = np.concatenate([mel, melt], -1)
            lf0 = np.concatenate([lf0, lf0t], 0)
        zero_idxs = np.where(lf0 == 0.0)[0]
        nonzero_idxs = np.where(lf0 != 0.0)[0]
        if len(nonzero_idxs) > 0 :
            mean = np.mean(lf0[nonzero_idxs])
            std = np.std(lf0[nonzero_idxs])
            if std == 0:
                lf0 -= mean
                lf0[zero_idxs] = 0.0
            else:
                lf0 = (lf0 - mean) / (std + 1e-8)
                lf0[zero_idxs] = 0.0

        pos = random.randint(0, mel.shape[-1] - self.n_sample_frames)
        mel = mel[:, pos:pos + self.n_sample_frames]
        lf0 = lf0[pos:pos + self.n_sample_frames]
        return torch.from_numpy(mel), torch.from_numpy(lf0)


class ContinuumDateset(Dataset):
    '''A Dataset contain f0 and spectrogram.

        Args:
            f0_dir, the path which contain f0 npy-file.
            spec_dir, the path which contain lin-spec npy-file.
    '''
    def __init__(self, spec_dir, f0_dir):
        self.n_frames = 64
        self.f0_dir = f0_dir
        self.spec_dir = spec_dir
        self.f0_lst = os.listdir(f0_dir)
        self.spec_lst = os.listdir(spec_dir)

    def get_f0_spec(self, lf0_path, spec_path):
        lf0 = np.load(lf0_path)
        lf0 = log(lf0)
        mel = np.load(spec_path).T
        melt = mel
        lf0t = lf0
        while mel.shape[-1] <= self.n_frames:
            mel = np.concatenate([mel, melt], -1)
            lf0 = np.concatenate([lf0, lf0t], 0)
        # print(mel.shape)
        zero_idxs = np.where(lf0 == 0.0)[0]
        nonzero_idxs = np.where(lf0 != 0.0)[0]
        if len(nonzero_idxs) > 0 :
            mean = np.mean(lf0[nonzero_idxs])
            std = np.std(lf0[nonzero_idxs])
            if std == 0:
                lf0 -= mean
                lf0[zero_idxs] = 0.0
            else:
                lf0 = (lf0 - mean) / (std + 1e-8)
                lf0[zero_idxs] = 0.0
        pos = np.random.randint(0, mel.shape[-1] - self.n_frames)
        mel = mel[:, pos:pos+self.n_frames]
        lf0 = lf0[pos:pos+self.n_frames]
        return torch.from_numpy(mel).float(), torch.from_numpy(lf0).float()

    def __getitem__(self, index):
        lf0_path = os.path.join(self.f0_dir, self.f0_lst[index])
        spec_path = os.path.join(self.spec_dir, self.spec_lst[index])
        spec, lf0 = self.get_f0_spec(lf0_path, spec_path)
        return spec, lf0

    def __len__(self):
        return len(self.spec_lst)


class ContinuumMultiDateset(Dataset):
    '''A Dataset contain f0 and spectrogram.

        Args:
            f0_dir, the path which contain f0 npy-file.
            spec_dir, the path which contain lin-spec npy-file.
    '''
    def __init__(self, spec_dir, F1_dir, F2_dir):
        self.n_frames = 64
        self.F1_dir = F1_dir
        self.F2_dir = F2_dir
        self.spec_dir = spec_dir
        self.F1_lst = os.listdir(F1_dir)
        self.F2_lst = os.listdir(F2_dir)
        self.spec_lst = os.listdir(spec_dir)

    def get_f0_spec(self, F1_path, F2_path, spec_path):
        F1 = np.load(F1_path)
        lF1 = log(F1)
        F2 = np.load(F2_path)
        lF2 = log(F2)
        mel = np.load(spec_path).T
        melt = mel
        lF1t = lF1
        lF2t = lF2
        while mel.shape[-1] <= self.n_frames:
            mel = np.concatenate([mel, melt], -1)
            lF1 = np.concatenate([lF1, lF1t], 0)
            lF2 = np.concatenate([lF2, lF2t], 0)
        # print(mel.shape)
        zero_idxs = np.where(lF1 == 0.0)[0]
        nonzero_idxs = np.where(lF1 != 0.0)[0]
        if len(nonzero_idxs) > 0 :
            mean = np.mean(lF1[nonzero_idxs])
            std = np.std(lF1[nonzero_idxs])
            if std == 0:
                lF1 -= mean
                lF1[zero_idxs] = 0.0
            else:
                lF1 = (lF1 - mean) / (std + 1e-8)
                lF1[zero_idxs] = 0.0
        zero_idxs = np.where(lF1 == 0.0)[0]
        # normliza lF2
        nonzero_idxs = np.where(lF2 != 0.0)[0]
        if len(nonzero_idxs) > 0 :
            mean = np.mean(lF2[nonzero_idxs])
            std = np.std(lF2[nonzero_idxs])
            if std == 0:
                lF2 -= mean
                lF2[zero_idxs] = 0.0
            else:
                lF2 = (lF2 - mean) / (std + 1e-8)
                lF2[zero_idxs] = 0.0

        pos = np.random.randint(0, mel.shape[-1] - self.n_frames)
        mel = mel[:, pos:pos+self.n_frames]
        lF1 = lF1[pos:pos+self.n_frames]
        lF2 = lF2[pos:pos+self.n_frames]
        return torch.from_numpy(mel).float(), torch.from_numpy(lF1).float(), torch.from_numpy(lF2).float()

    def __getitem__(self, index):
        F1_path = os.path.join(self.F1_dir, self.F1_lst[index])
        F2_path = os.path.join(self.F2_dir, self.F2_lst[index])
        spec_path = os.path.join(self.spec_dir, self.spec_lst[index])
        spec, lF1, lF2 = self.get_f0_spec(F1_path, F2_path, spec_path)
        return spec, lF1, lF2

    def __len__(self):
        return len(self.spec_lst)
