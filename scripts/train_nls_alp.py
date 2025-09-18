#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normal Latent Synthesis (NLS) + Anomaly Latent Perturbation (ALP)
- NLS: train Conv Encoder/Decoder on log-mel with SSIM (+L1)
- FCAE: train fully-connected AE on latents (last-dim)
- ALP: gradient ascent on a detached latent; project with FCAE; decode with frozen decoder
- Outputs: synthesized normal mel + synthesized anomaly mel
"""

import os, glob, math, argparse, warnings
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_function
from torch.utils.data import Dataset, DataLoader

import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- your utilities ---
from sans.audio_utils import WaveToMel
from sans.fc_autoencoder import FcAutoEncoder, FcAeConfig  # your earlier file

# =========================================================
# Data
# =========================================================
class AudioFolder(Dataset):
    def __init__(self, root: str, sr: int = 16000, duration_s: float = 2.0, exts=(".wav", ".flac", ".mp3")):
        self.paths = []
        for e in exts:
            self.paths += glob.glob(os.path.join(root, f"**/*{e}"), recursive=True)
        if not self.paths:
            raise RuntimeError(f"No audio found under {root}")
        self.sr = sr
        self.samples = int(sr * duration_s)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        wav, _ = librosa.load(p, sr=self.sr, mono=True)
        if len(wav) < self.samples:
            pad = self.samples - len(wav)
            wav = np.pad(wav, (0, pad))
        else:
            wav = wav[:self.samples]
        return torch.tensor(wav, dtype=torch.float32)

# =========================================================
# SSIM for 2D (spectrograms)
# =========================================================
class SSIM(nn.Module):
    def __init__(self, win=11, sigma=1.5, channel=1):
        super().__init__()
        gauss = torch.tensor([math.exp(-(x - win//2)**2/(2*sigma**2)) for x in range(win)], dtype=torch.float32)
        gauss = (gauss / gauss.sum()).unsqueeze(0)  # [1, win]
        kernel = (gauss.t() @ gauss).unsqueeze(0).unsqueeze(0)  # [1,1,win,win]
        self.register_buffer("kernel", kernel)
        self.C1 = 0.01**2
        self.C2 = 0.03**2
        self.win = win

    def forward(self, x, y):
        # x,y: [B,1,F,T]
        pad = self.win // 2
        mu_x = torch_function.conv2d(x, self.kernel, padding=pad)
        mu_y = torch_function.conv2d(y, self.kernel, padding=pad)
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y
        sigma_x2 = torch_function.conv2d(x*x, self.kernel, padding=pad) - mu_x2
        sigma_y2 = torch_function.conv2d(y*y, self.kernel, padding=pad) - mu_y2
        sigma_xy = torch_function.conv2d(x*y, self.kernel, padding=pad) - mu_xy
        ssim_num = (2*mu_xy + self.C1) * (2*sigma_xy + self.C2)
        ssim_den = (mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2)
        ssim_map = ssim_num / (ssim_den + 1e-12)
        return ssim_map.mean()

# =========================================================
# Conv Encoder/Decoder (latent <-> mel) for NLS/ALP
# =========================================================
class ConvEncoder(nn.Module):
    def __init__(self, in_ch=1, lat_ch=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, lat_ch, 1)
        )
    def forward(self, x):  # [B,1,F,T] -> [B,C,F,T]
        return self.net(x)

class ConvDecoder(nn.Module):
    def __init__(self, lat_ch=8, out_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(lat_ch, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, out_ch, 1)
        )
    def forward(self, z):  # [B,C,F,T] -> [B,1,F,T]
        return self.net(z)

# =========================================================
# Helpers
# =========================================================
def rms_dbfs(x: torch.Tensor, eps=1e-12):
    if x.ndim == 1: x = x.unsqueeze(0)
    rms = torch.sqrt((x**2).mean(dim=-1, keepdim=True) + eps)
    return 20.0*torch.log10(rms.clamp_min(eps))

def normalize_rms(x: torch.Tensor, target_dbfs=-20.0, peak_clip=0.999):
    if x.ndim == 1: x = x.unsqueeze(0)
    gain = 10.0 ** ((target_dbfs - rms_dbfs(x)) / 20.0)
    y = x * gain
    return y.clamp_(-peak_clip, peak_clip)

def ensure_mel_b1ft(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4: 
        if x.size(1) != 1: x = x.mean(1, keepdim=True)
        return x
    if x.ndim == 3:
        B,A,B2 = x.shape
        return x.unsqueeze(1) if A <= B2 else x.transpose(1,2).unsqueeze(1)
    if x.ndim == 2:
        F,T = x.shape
        return x.unsqueeze(0).unsqueeze(0) if F <= T else x.t().unsqueeze(0).unsqueeze(0)
    if x.ndim == 1:
        return x.unsqueeze(0).unsqueeze(0)
    raise RuntimeError(f"bad shape {tuple(x.shape)}")

def save_mel_time_y(mel_b1ft: torch.Tensor, path: str, sr: int, hop: int, title=None, vmin=None, vmax=None):
    m = ensure_mel_b1ft(mel_b1ft).detach().cpu().numpy()[0,0]
    img = m.T  # time on Y
    t_max = (img.shape[0]-1) * (hop/sr)
    if vmin is None or vmax is None:
        vmin = float(np.percentile(img, 2.0)); vmax = float(np.percentile(img, 98.0))
    plt.figure(figsize=(6,6))
    plt.imshow(img, aspect="auto", origin="lower", extent=[0, m.shape[0]-1, 0, t_max], vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(); cbar.set_label("log-mel")
    plt.xlabel("Mel bins"); plt.ylabel("Time (s)")
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# =========================================================
# Training / ALP
# =========================================================
def train_nls(
    enc: nn.Module, dec: nn.Module, dl: DataLoader, to_mel: WaveToMel,
    device: str, epochs: int = 5, lr: float = 1e-3, lambda_l1: float = 0.1
):
    ssim = SSIM().to(device)
    opt = torch.optim.Adam(list(enc.parameters())+list(dec.parameters()), lr=lr)
    enc.train(); dec.train()
    for ep in range(1, epochs+1):
        pbar = tqdm(dl, desc=f"[NLS] epoch {ep}/{epochs}")
        for wav in pbar:
            wav = wav.to(device)
            mel = ensure_mel_b1ft(to_mel(wav))          # [B,1,F,T]
            mel = mel / (mel.abs().max(dim=-1,keepdim=True)[0].clamp_min(1e-6))  # basic scale
            z = enc(mel)                                 # [B,C,F,T]
            mel_hat = dec(z)                             # [B,1,F,T]
            ssim_val = ssim(mel_hat, mel)                # higher is better
            l1 = torch_function.l1_loss(mel_hat, mel)
            loss = (1.0 - ssim_val) + lambda_l1 * l1
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(ssim=float(ssim_val), l1=float(l1))
    enc.eval(); dec.eval()

def train_fcae_on_latents(
    enc: nn.Module, fcae: FcAutoEncoder, dl: DataLoader, to_mel: WaveToMel,
    device: str, epochs: int = 3, lr: float = 1e-3
):
    opt = torch.optim.Adam(fcae.parameters(), lr=lr)
    fcae.train(); enc.eval()
    for ep in range(1, epochs+1):
        pbar = tqdm(dl, desc=f"[FCAE] epoch {ep}/{epochs}")
        for wav in pbar:
            wav = wav.to(device)
            mel = ensure_mel_b1ft(to_mel(wav))
            z = enc(mel)                                 # [B,C,F,T]
            # treat last dim as time; collapse channel*freq into batch for FcAE:
            B,C,F,T = z.shape
            z2 = z.permute(0,2,1,3).reshape(B*F, C, T)   # [B*F, C, T]
            # FCAE works on last dim; put (B*F*C, T):
            z2 = z2.reshape(B*F*C, T)                    # [N, T]
            recon = fcae(z2)                             # [N, T]
            loss = torch_function.mse_loss(recon, z2)
            opt.zero_grad(); loss.backward(); opt.step()
            pbar.set_postfix(mse=float(loss))
    fcae.eval()

@torch.no_grad()
def synth_normal(enc, dec, wav_batch, to_mel, device):
    mel = ensure_mel_b1ft(to_mel(wav_batch.to(device)))
    z = enc(mel)
    mel_hat = dec(z)
    return mel_hat

def alp_generate(
    enc: nn.Module, dec: nn.Module, fcae: FcAutoEncoder,
    wav: torch.Tensor, to_mel: WaveToMel,
    steps: int = 25, lr: float = 5e-2, tau: float = 2.0,
    band_obj: Tuple[int,int] | None = None,   # e.g., (64,127)
    device: str = "cuda"
):
    """
    Gradient AScent on detached latent, project via FCAE at each step, decode with frozen decoder.
    """
    ssim = SSIM().to(device)
    with torch.no_grad():
        mel = ensure_mel_b1ft(to_mel(wav.to(device)))   # [1,1,F,T]
        z0  = enc(mel)                                  # [1,C,F,T]
    z = z0.detach().clone().requires_grad_(True)

    # objective: (1 - SSIM(dec(z), mel)) + optional band energy
    def objective(z):
        mel_hat = dec(z)
        loss = 1.0 - ssim(mel_hat, mel)                 # want dissimilarity
        if band_obj is not None:
            lo,hi = band_obj
            mh = mel_hat[:, :, lo:hi+1, :]              # [B,1,band,T]
            loss = loss + 0.1 * (-mh.mean())            # encourage energy in band
        return loss

    opt = torch.optim.SGD([z], lr=lr)
    for _ in tqdm(range(steps), desc="[ALP] ascent"):
        loss = objective(z)
        opt.zero_grad(); (-loss).backward(); opt.step()  # gradient ASCENT

        # radius control around z0
        with torch.no_grad():
            d = z - z0
            n = d.pow(2).mean().sqrt().clamp_min(1e-6)
            z.copy_(z0 + d * (tau / max(tau, float(n))))  # simple norm clamp

        # projection with FCAE (detach & project last dim)
        with torch.no_grad():
            B,C,F,T = z.shape
            zz = z.permute(0,2,1,3).reshape(B*F, C, T).reshape(B*F*C, T)  # [N,T]
            zz_proj = fcae(zz)                                            # [N,T]
            zz_proj = zz_proj.view(B*F, C, T).reshape(B, F, C, T).permute(0,2,1,3)  # [B,C,F,T]
            z.copy_(zz_proj)

        z.requires_grad_(True)

    with torch.no_grad():
        mel_anom = dec(z)
    return mel_anom, z0, z

# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="folder of *normal* audio")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--hop", type=int, default=80)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs_nls", type=int, default=5)
    ap.add_argument("--epochs_fcae", type=int, default=3)
    ap.add_argument("--lat_ch", type=int, default=8)
    ap.add_argument("--fcae_bottleneck", type=int, default=256)
    ap.add_argument("--ascent_steps", type=int, default=25)
    ap.add_argument("--ascent_lr", type=float, default=5e-2)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--band_lo", type=int, default=64)
    ap.add_argument("--band_hi", type=int, default=127)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__} | CUDA={torch.cuda.is_available()} | device={device}")

    ds = AudioFolder(args.data, sr=args.sr, duration_s=args.duration)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    to_mel = WaveToMel(sr=args.sr, hop=args.hop, n_mels=args.n_mels).to(device)

    # --- models ---
    enc = ConvEncoder(in_ch=1, lat_ch=args.lat_ch).to(device)
    dec = ConvDecoder(lat_ch=args.lat_ch, out_ch=1).to(device)
    fcae = FcAutoEncoder(FcAeConfig(in_dim=int(args.duration*args.sr/args.hop),  # latent T length ≈ frames
                                    bottleneck=args.fcae_bottleneck,
                                    hidden_mult=2.0, depth=2, dropout=0.0,
                                    activation="gelu", layernorm=True)).to(device)

    # --- stage 1: NLS (train enc/dec) ---
    train_nls(enc, dec, dl, to_mel, device, epochs=args.epochs_nls, lr=1e-3, lambda_l1=0.1)

    # --- stage 2: FCAE (train on latents) ---
    train_fcae_on_latents(enc, fcae, dl, to_mel, device, epochs=args.epochs_fcae, lr=1e-3)

    # --- demo: synthesize one batch normal & anomaly ---
    wav_batch = next(iter(dl)).to(device)
    wav_norm = normalize_rms(wav_batch, target_dbfs=-20.0)
    mel_normal = synth_normal(enc, dec, wav_norm, to_mel, device)

    # ALP on first item for visualization
    mel_anom, z0, zA = alp_generate(
        enc, dec, fcae,
        wav_norm[:1], to_mel,
        steps=args.ascent_steps, lr=args.ascent_lr, tau=args.tau,
        band_obj=(args.band_lo, args.band_hi), device=device
    )

    # --- save figures (time on Y) ---
    save_mel_time_y(ensure_mel_b1ft(to_mel(wav_norm[:1])), "original_norm.png", args.sr, args.hop, "Original (RMS-norm)")
    save_mel_time_y(mel_normal[:1], "synth_normal.png", args.sr, args.hop, "Synthesized Normal (NLS)")
    save_mel_time_y(mel_anom, "synth_anomaly.png", args.sr, args.hop, "Synthesized Anomaly (ALP)")

    # --- save checkpoints ---
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict()}, "checkpoints/conv_ae.pt")
    fcae.save("checkpoints/fc_ae.pt")
    torch.save({"z0": z0.cpu(), "zA": zA.cpu(), "mel_normal": mel_normal.cpu(), "mel_anom": mel_anom.cpu()},
               "checkpoints/example_outputs.pt")

    print("Wrote: original_norm.png, synth_normal.png, synth_anomaly.png and checkpoints/")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
