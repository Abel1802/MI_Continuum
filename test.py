import numpy as np
import os
import subprocess
import soundfile as sf
import pyworld as pw
import kaldiio
import torch 
import matplotlib.pyplot as plt 

from model import Encoder
from model import Encoder_f0
from model import Decoder
from spectrogram import logmelspectrogram


def extract_logmel(wav_path, mean, std, sr=16000):
    # wav, fs = librosa.load(wav_path, sr=sr)
    wav, fs = sf.read(wav_path)
    if fs != sr:
        wav = resampy.resample(wav, fs, sr, axis=0)
        fs = sr
    # wav, _ = librosa.effects.trim(wav, top_db=15)
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
        window="hann",
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
    exp_name = "F001_mel_f0_with_MINE_0.5_mi_loss"
    resume_epoch = 500
    checkpoint_path = f"exp/{exp_name}/checkpoints/model.ckpt-{resume_epoch}.pt"
    mel_stats = np.load("/disk2/lz/workspace/data_new/mel_stats.npy")
    mel_mean, mel_std = mel_stats[0], mel_stats[1]
    out_dir = f"exp/{exp_name}/converted/"
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device('cuda:1')

    # test_data
    feat_writer = kaldiio.WriteHelper(
            "ark,scp:{o}.ark,{o}.scp".format(o=str(out_dir) + "/feats.1")
        )
    
    src_wav_path = "/disk2/lz/workspace/data_new/test_wavs/bei1.wav"
    ref_wav_path = "/disk2/lz/workspace/data_new/test_wavs/bei4_fix.wav"
    out_filename = os.path.basename(src_wav_path).split(".")[0]
    src_mel, src_lf0 = extract_logmel(src_wav_path, mel_mean, mel_std)
    ref_mel, ref_lf0 = extract_logmel(ref_wav_path, mel_mean, mel_std)

    src_mel = torch.FloatTensor(src_mel.T).unsqueeze(0).to(device)
    src_lf0 = torch.FloatTensor(src_lf0).unsqueeze(0).to(device)
    ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)
    ref_lf0 = torch.FloatTensor(ref_lf0).unsqueeze(0).to(device)

    # load trained model
    encoder_f0 = Encoder_f0().to(device)
    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    checkpoint = torch.load(checkpoint_path)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder_f0.load_state_dict(checkpoint['encoder_f0'])
    decoder.load_state_dict(checkpoint['decoder'])

    # inference
    with torch.no_grad():
        src_lf0_emb = encoder_f0(src_lf0)
        ref_lf0_emb = encoder_f0(ref_lf0)

        z_src_mel = encoder(src_mel)
        z_ref_mel = encoder(ref_mel)

        _, pred_src_mel = decoder(z_src_mel, src_lf0_emb)
        _, pred_src_mel_ref_lf0 = decoder(z_src_mel, ref_lf0_emb)
        _, pred_ref_mel_src_lf0 = decoder(z_ref_mel, src_lf0_emb)
        _, pred_ref_mel = decoder(z_ref_mel, ref_lf0_emb)

        # Plot mel
        plt.figure(figsize=(10, 8))
        plt.subplot(141)
        plt.imshow(src_mel.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(142)
        plt.imshow(pred_src_mel.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(143)
        plt.imshow(pred_src_mel_ref_lf0.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.subplot(144)
        plt.imshow(ref_mel.squeeze(0).cpu().numpy(), origin='lower', aspect='auto')
        plt.savefig(f'exp/{exp_name}/resume_{resume_epoch}.png')

        feat_writer[out_filename + "_converted"] = pred_src_mel.squeeze(0).cpu().numpy()
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