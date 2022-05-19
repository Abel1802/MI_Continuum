import numpy as np
import os
import subprocess
import soundfile as sf
import resampy
import librosa
import pyworld as pw
import kaldiio
import torch 
import matplotlib.pyplot as plt 

from model import Encoder
from model import Encoder_f0
from model import DecoderMulti
from spectrogram import logmelspectrogram
from utils import get_formant
from utils import norm_f0


def extract_logmel(wav_path, mean, std, sr=16000):
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
                n_fft=400,
                n_shift=160,
                win_length=400,
                window='hann',
                fmin=80,
                fmax=7600,
            )
    mel = (mel - mean) / (std + 1e-8)
    tlen = mel.shape[0]
    frame_period = 160 / fs * 1000
    f0, timeaxis = pw.dio(wav.astype("float64"), fs, frame_period=frame_period)
    f0 = pw.stonemask(wav.astype("float64"), f0, timeaxis, fs)
    f0 = f0[:tlen].reshape(-1).astype("float32")
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(
        f0[nonzeros_indices]
    )  # for f0(Hz), lf0 > 0 when f0 != 0
    mean, std = np.mean(lf0[nonzeros_indices]), np.std(lf0[nonzeros_indices])
    lf0[nonzeros_indices] = (lf0[nonzeros_indices] - mean) / (std + 1e-8)
    return mel, lf0


def main():
    # checkpoint_path
    resume_epoch = 500
    exp_name = "F001_mel_F1_F2_with_MINE_100_mi_loss"
    checkpoint_path = f"exp/{exp_name}/checkpoints/model.ckpt-{resume_epoch}.pt"
    mel_stats = np.load("/disk2/lz/workspace/data_new/F001/mel_stats.npy")
    mel_mean, mel_std = mel_stats[0], mel_stats[1]
    out_dir = f"exp/{exp_name}/converted/"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda:0')

    # test_data
    feat_writer = kaldiio.WriteHelper(
            "ark,scp:{o}.ark,{o}.scp".format(o=str(out_dir) + "/feats.1")
        )
    
    src_wav_path = "/disk2/lz/workspace/data_new/test_wavs/a1.wav"
    ref_wav_path = "/disk2/lz/workspace/data_new/test_wavs/yi1_fix.wav"
    out_filename = os.path.basename(src_wav_path).split(".")[0]
    src_mel, src_lf0 = extract_logmel(src_wav_path, mel_mean, mel_std)
    ref_mel, ref_lf0 = extract_logmel(ref_wav_path, mel_mean, mel_std)

    # get formant
    src_lF1 = get_formant(src_wav_path, src_lf0, formant_num=1)
    src_lF1 = norm_f0(np.array(src_lF1))
    src_lF2 = get_formant(src_wav_path, src_lf0, formant_num=2)
    src_lF2 = norm_f0(np.array(src_lF2))

    ref_lF1 = get_formant(ref_wav_path, ref_lf0, formant_num=1)
    ref_lF1 = norm_f0(np.array(ref_lF1))
    ref_lF2 = get_formant(ref_wav_path, ref_lf0, formant_num=2)
    ref_lF2 = norm_f0(np.array(ref_lF2))

    src_mel = torch.FloatTensor(src_mel.T).unsqueeze(0).to(device)
    src_lF1 = torch.FloatTensor(src_lF1).unsqueeze(0).to(device)
    src_lF2 = torch.FloatTensor(src_lF2).unsqueeze(0).to(device)
    ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)
    ref_lF1 = torch.FloatTensor(ref_lF1).unsqueeze(0).to(device)
    ref_lF2 = torch.FloatTensor(ref_lF2).unsqueeze(0).to(device)

    # load trained model
    encoder_F1 = Encoder_f0().to(device)
    encoder_F2 = Encoder_f0().to(device)
    encoder = Encoder().to(device)
    decoder = DecoderMulti().to(device)

    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder_F1.load_state_dict(checkpoint['encoder_F1'])
    encoder_F2.load_state_dict(checkpoint['encoder_F2'])
    decoder.load_state_dict(checkpoint['decoder'])

    encoder_F1.eval()
    encoder_F2.eval()
    encoder.eval()
    decoder.eval()

    # inference
    with torch.no_grad():
        src_lF1_emb = encoder_F1(src_lF1)
        src_lF2_emb = encoder_F2(src_lF2)
        ref_lF1_emb = encoder_F1(ref_lF1)
        ref_lF2_emb = encoder_F2(ref_lF2)

        z_src_mel = encoder(src_mel)
        z_ref_mel = encoder(ref_mel)

        _, pred_src_mel = decoder(z_src_mel, src_lF1_emb, src_lF2_emb)
        _, pred_src_mel_ref_lF1 = decoder(z_src_mel, ref_lF1_emb, src_lF2_emb)
        _, pred_src_mel_ref_lF2 = decoder(z_src_mel, src_lF1_emb, ref_lF2_emb)
        _, pred_src_mel_ref_lF1_lF2 = decoder(z_src_mel, ref_lF1_emb, ref_lF2_emb)
        _, pred_ref_mel = decoder(z_ref_mel, ref_lF1_emb, ref_lF2_emb)

        # continuum
        b_ints = np.linspace(0, 1, 10)
        lF1_ints = [src_lF1_emb * (1 - b) + ref_lF1_emb * b for b in b_ints ]
        lF2_ints = [src_lF2_emb * (1 - b) + ref_lF2_emb * b for b in b_ints ]
        mel_ints = [decoder(z_src_mel, lF1_int, lF2_int)[1] for lF1_int, lF2_int in zip(lF1_ints, lF2_ints)]
        for i in range(10):
            feat_writer[out_filename + f"continuum_{i}"] = mel_ints[i].squeeze(0).cpu().numpy().T

        # Plot mel
        plt.figure(figsize=(10, 8))
        plt.subplot(161)
        plt.imshow(src_mel.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(162)
        plt.imshow(pred_src_mel.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(163)
        plt.imshow(pred_src_mel_ref_lF1.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(164)
        plt.imshow(pred_src_mel_ref_lF2.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(165)
        plt.imshow(pred_src_mel_ref_lF1_lF2.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(166)
        plt.imshow(ref_mel.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.savefig(f'exp/{exp_name}/resume_{resume_epoch}.png')

        feat_writer[out_filename + "F1_converted"] = pred_src_mel_ref_lF1.squeeze(0).cpu().numpy().T
        feat_writer[out_filename + "F2_converted"] = pred_src_mel_ref_lF2.squeeze(0).cpu().numpy().T
        feat_writer[out_filename + "F1_F2_converted"] = pred_src_mel_ref_lF1_lF2.squeeze(0).cpu().numpy().T
        feat_writer[out_filename + "_source"] = src_mel.squeeze(0).cpu().numpy().T
        feat_writer[out_filename + "_reference"] = (
            ref_mel.squeeze(0).cpu().numpy().T
        )
    feat_writer.close()

    # vocoder
    print("synthesize waveform...")
    cmd = [
        "parallel-wavegan-decode",
        "--checkpoint",
        "./vocoder/checkpoint-3000000steps.pkl",
        "--feats-scp",
        f"{str(out_dir)}/feats.1.scp",
        "--outdir",
        str(out_dir),
    ]
    subprocess.call(cmd)

if __name__ == "__main__":
    main()
