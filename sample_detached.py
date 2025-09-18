#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, warnings, sys, types, inspect
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn, librosa
from tqdm import tqdm

from sans import build_model
from sans.audio_utils import WaveToMel
from sans.objectives import band_energy_objective
from sans.pipeline import style_transfer

AUDIO_PROMPT_TOKEN = "__AUDIO_CTX__"

# ---------------- utils ----------------
def rms_dbfs(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if x.ndim == 1: x = x.unsqueeze(0)
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
    return 20.0 * torch.log10(rms.clamp_min(eps))

def normalize_rms(x: torch.Tensor, target_dbfs: float = -20.0, peak_clip: float = 0.999) -> torch.Tensor:
    if x.ndim == 1: x = x.unsqueeze(0)
    gain = 10.0 ** ((target_dbfs - rms_dbfs(x)) / 20.0)
    y = x * gain
    return y.clamp_(-peak_clip, peak_clip)

def ensure_mel_b1ft(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    t = x
    if t.ndim == 4:
        if t.size(1) != 1: t = t.mean(dim=1, keepdim=True)
        return t.contiguous()
    if t.ndim == 3:
        B, A, B2 = t.shape
        return (t.unsqueeze(1) if A <= B2 else t.transpose(1,2).unsqueeze(1)).contiguous()
    if t.ndim == 2:
        F, T = t.shape
        return (t.unsqueeze(0).unsqueeze(0) if F <= T else t.t().unsqueeze(0).unsqueeze(0)).contiguous()
    if t.ndim == 1:
        return t.unsqueeze(0).unsqueeze(0).contiguous()
    raise RuntimeError(f"Unsupported: {tuple(x.shape)}")

def to_mel_or_passthrough(x: torch.Tensor, to_mel: nn.Module) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 1: return ensure_mel_b1ft(to_mel(x.unsqueeze(0)))
    if x.ndim == 2 and x.shape[0] <= 4: return ensure_mel_b1ft(to_mel(x))
    if x.ndim == 3 and x.shape[1] == 1 and x.shape[2] > 8: return ensure_mel_b1ft(to_mel(x.squeeze(1)))
    return ensure_mel_b1ft(x)

def plot_mel_time_y(mel_b1ft: torch.Tensor, *, path: str, sr: int, hop: int, title: str = None, vmin=None, vmax=None):
    m = ensure_mel_b1ft(mel_b1ft).detach().cpu().numpy()
    img = m[0, 0]
    F, T = img.shape
    data = img.T
    t_y = (T - 1) * (hop / sr)
    if vmin is None or vmax is None:
        vmin = float(np.percentile(data, 2.0)); vmax = float(np.percentile(data, 98.0))
    extent = [0.0, float(F - 1), 0.0, t_y]
    plt.figure(figsize=(6, 6))
    plt.imshow(data, aspect="auto", origin="lower", extent=extent, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(); cbar.set_label("log-mel")
    plt.xlabel("Mel bins"); plt.ylabel("Time (s)")
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# ---------------- audio->ctx projection (uses context width from LDM) ----------
def _text_ctx_shape(ldm):
    dummy = ldm.get_learned_conditioning([""])
    return int(dummy.shape[1]), int(dummy.shape[2])  # (N_tokens, D_ctx)

def _ensure_mel_b1ft_local(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if x.ndim == 4:
        if x.size(1) != 1: x = x.mean(dim=1, keepdim=True)
        return x.contiguous()
    if x.ndim == 3:
        B, A, B2 = x.shape
        return (x.unsqueeze(1) if A <= B2 else x.transpose(1,2).unsqueeze(1)).contiguous()
    if x.ndim == 2:
        F, T = x.shape
        return (x.unsqueeze(0).unsqueeze(0) if F <= T else x.t().unsqueeze(0).unsqueeze(0)).contiguous()
    if x.ndim == 1:
        return x.unsqueeze(0).unsqueeze(0).contiguous()
    raise RuntimeError(f"Unsupported: {tuple(x.shape)}")

def _make_audio_context(ldm, e: torch.Tensor) -> torch.Tensor:
    # e -> [B,T] then project to text ctx dims [B,N,D]
    if e.ndim >= 3:
        mel = _ensure_mel_b1ft_local(e).squeeze(1)  # [B,F,T]
        vec = mel.mean(dim=1)                       # [B,T]
    elif e.ndim == 2:
        vec = e                                     # [B,T’]
    else:
        vec = e.unsqueeze(0)                        # [1,T’]

    N, D = _text_ctx_shape(ldm)
    B, Tin = vec.shape
    dev = vec.device

    key = "_audio2ctx_proj"
    if not hasattr(ldm, key):
        setattr(ldm, key, nn.Linear(Tin, D, bias=True).to(dev))
        nn.init.xavier_uniform_(getattr(ldm, key).weight, gain=0.5)
        nn.init.zeros_(getattr(ldm, key).bias)
    proj: nn.Linear = getattr(ldm, key)
    if proj.in_features != Tin or proj.out_features != D:
        new_proj = nn.Linear(Tin, D, bias=True).to(dev)
        nn.init.xavier_uniform_(new_proj.weight, gain=0.5)
        nn.init.zeros_(new_proj.bias)
        setattr(ldm, key, new_proj)
        proj = new_proj

    ctx1 = proj(vec)             # [B,D]
    ctx  = ctx1.unsqueeze(1)     # [B,1,D]
    if N > 1:
        ctx = ctx.expand(B, N, D).contiguous()
    return ctx

# ---------------- HARD injection at sampler + UNet; x-bias fallback -----------
class SamplerCtxPatcher:
    """
    Patch sampler entry (model.apply_model) and UNet.forward to force audio ctx AND
    optionally add a tiny audio-dependent bias to x (so conditioning has guaranteed effect).
    """
    def __init__(self, ldm, ctx: torch.Tensor, x_bias: float = 0.05, log_first: bool = True):
        self.ldm = ldm
        self.ctx = ctx
        self.x_bias = float(x_bias)
        self._apply_model_orig = None
        self._unet = None
        self._unet_fwd_orig = None
        self._glc_orig = None
        self._xproj = None
        self._printed = not log_first  # print only once

    def __enter__(self):
        ldm, ctx = self.ldm, self.ctx

        # Locate UNet-ish module
        unet = None
        for name in ("model", "diffusion_model"):
            if hasattr(ldm, name):
                cand = getattr(ldm, name)
                if hasattr(cand, "forward"):
                    unet = cand
                    break
        if unet is None and hasattr(ldm, "model") and hasattr(ldm.model, "diffusion_model"):
            unet = ldm.model.diffusion_model
        self._unet = unet

        # Patch model.apply_model (sampler entry) if present
        model = getattr(ldm, "model", ldm)
        if hasattr(model, "apply_model"):
            self._apply_model_orig = model.apply_model

            def make_apply_wrapped(orig):
                def wrapped(x, t, cond=None, *args, **kwargs):
                    # ---- inject cond ----
                    if cond is None: cond = {}
                    if isinstance(cond, dict):
                        c = dict(cond); c["c_crossattn"] = [ctx]; cond = c
                    kwargs["context"] = ctx
                    kwargs["encoder_hidden_states"] = ctx

                    # ---- x-bias fallback (guaranteed effect) ----
                    xb = x
                    if self.x_bias > 0.0 and x.dim() == 4:
                        B, C, H, W = x.shape
                        Bc, N, D = ctx.shape
                        dev = x.device
                        # avg over tokens -> [B,D]
                        cavg = ctx.mean(dim=1)
                        # build / reuse linear D->C
                        if self._xproj is None or self._xproj.in_features != D or self._xproj.out_features != C:
                            self._xproj = nn.Linear(D, C, bias=True).to(dev)
                            nn.init.zeros_(self._xproj.bias)
                            nn.init.normal_(self._xproj.weight, std=0.02)
                        bias = self._xproj(cavg).view(B, C, 1, 1)
                        xb = x + self.x_bias * bias
                    # ---- first-call debug ----
                    if not self._printed:
                        self._printed = True
                        print("[inject] Patched model.apply_model",
                              f"x={tuple(x.shape)} -> {tuple(xb.shape)}  t={tuple(t.shape) if torch.is_tensor(t) else t}",
                              f"cond_keys={list(cond.keys()) if isinstance(cond, dict) else type(cond)}",
                              f"kwargs_keys={list(kwargs.keys())}", sep="\n", flush=True)
                    return orig(xb, t, cond, *args, **kwargs)
                return wrapped

            model.apply_model = make_apply_wrapped(self._apply_model_orig)
            print("[inject] model.apply_model: forcing c_crossattn/context + x-bias", flush=True)

        # Patch UNet.forward (safety net for other paths)
        if self._unet is not None:
            self._unet_fwd_orig = self._unet.forward
            def make_unet_wrapped(orig_forward):
                def wrapped_forward(*args, **kwargs):
                    if "context" in kwargs: kwargs["context"] = ctx
                    if "encoder_hidden_states" in kwargs: kwargs["encoder_hidden_states"] = ctx
                    if len(args) >= 3 and isinstance(args[2], dict):
                        cdict = dict(args[2]); cdict["c_crossattn"] = [ctx]
                        args = (args[0], args[1], cdict, *args[3:])
                    return orig_forward(*args, **kwargs)
                return wrapped_forward
            self._unet.forward = make_unet_wrapped(self._unet_fwd_orig)
            print("[inject] UNet.forward: stuffing context/cond", flush=True)

        # Patch get_learned_conditioning for our token (harmless)
        self._glc_orig = ldm.get_learned_conditioning
        def _glc_override(prompts):
            B = len(prompts) if isinstance(prompts, (list,tuple)) else 1
            use_audio = any(str(p) == AUDIO_PROMPT_TOKEN for p in (prompts if isinstance(prompts,(list,tuple)) else [prompts]))
            if use_audio:
                N, D = _text_ctx_shape(ldm)
                return torch.zeros(B, N, D, device=ctx.device, dtype=ctx.dtype)
            return self._glc_orig(prompts)
        ldm.get_learned_conditioning = _glc_override

        return self

    def __exit__(self, exc_type, exc, tb):
        model = getattr(self.ldm, "model", self.ldm)
        if self._apply_model_orig is not None:
            model.apply_model = self._apply_model_orig
        if self._unet is not None and self._unet_fwd_orig is not None:
            self._unet.forward = self._unet_fwd_orig
        if self._glc_orig is not None:
            self.ldm.get_learned_conditioning = self._glc_orig

def force_ctx_generate(ldm, e, *, steps, guidance_scale, duration_s, seed, ref_path, sr, output_type="mel", x_bias=0.05):
    ctx = _make_audio_context(ldm, e)
    with SamplerCtxPatcher(ldm, ctx, x_bias=x_bias):
        out = style_transfer(
            ldm,
            AUDIO_PROMPT_TOKEN,   # we intercept this at cond stage (returns zeros), but sampler gets real ctx
            original_audio_file_path=ref_path,
            transfer_strength=0.0,
            duration=duration_s if duration_s is not None else 5.0,
            guidance_scale=guidance_scale,
            ddim_steps=steps,
            output_type=output_type,
            # sr=sr,
        )
    return out

# ---------------- Detached-AEFC (your idea; shape-agnostic on last dim) -------
class DetachedAEFC(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512, depth: int = 2,
                 dropout: float = 0.1, activation: str = "gelu",
                 layernorm: bool = True, alpha: float = 1.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.alpha = float(alpha)
        act = nn.GELU() if activation == "gelu" else (nn.SiLU() if activation == "silu" else nn.ReLU(inplace=True))
        H = max(16, int(hidden))
        layers = [nn.Linear(self.in_dim, H)]
        if layernorm: layers += [nn.LayerNorm(H)]
        layers += [act, nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(H, H)]
            if layernorm: layers += [nn.LayerNorm(H)]
            layers += [act, nn.Dropout(dropout)]
        layers += [nn.Linear(H, self.in_dim)]
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _repeat_to_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
        d = int(x.shape[-1])
        if d == target_dim: return x
        if d > target_dim:  return x[..., :target_dim].contiguous()
        reps = (target_dim + d - 1) // d
        xr = x.repeat_interleave(reps, dim=-1)
        return xr[..., :target_dim].contiguous()

    @staticmethod
    def _flatten_last(x: torch.Tensor):
        lead = tuple(x.shape[:-1]); D = int(x.shape[-1])
        N = 1
        for a in lead: N *= int(a)
        return x.reshape(N, D), lead, D

    @staticmethod
    def _unflatten_last(x2d: torch.Tensor, lead):
        return x2d.view(*lead, x2d.shape[-1])

    def forward(self, x: torch.Tensor, ref: torch.Tensor, alpha: float | None = None) -> torch.Tensor:
        D = int(x.shape[-1])
        xin  = self._repeat_to_dim(x.detach(),   self.in_dim)
        r_in = self._repeat_to_dim(ref.detach(), self.in_dim)
        x2d, lead, _ = self._flatten_last(xin)
        y2d = self.net(x2d)
        y   = self._unflatten_last(y2d, lead)
        y   = self._repeat_to_dim(y, D)
        r   = self._repeat_to_dim(r_in, D)
        a = (self.alpha if alpha is None else float(alpha))
        return r + a * (y - r)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="Force audio cond at sampler/UNet + tiny x-bias fallback; Detached-AEFC + ascent")
    ap.add_argument("--ckpt", type=str, default="/home/tori/.cache/audioldm/audioldm-s-full.ckpt")
    ap.add_argument("--ref", type=str, default="trumpet.wav")
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--hop", type=int, default=80)
    ap.add_argument("--n_mels", type=int, default=128)

    ap.add_argument("--ascent_steps", type=int, default=25)
    ap.add_argument("--inner_steps", type=int, default=12)
    ap.add_argument("--guidance", type=float, default=2.5)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--band_lo", type=int, default=64)
    ap.add_argument("--band_hi", type=int, default=127)
    ap.add_argument("--norm_dbfs", type=float, default=-20.0)

    ap.add_argument("--aefc_hidden", type=int, default=512)
    ap.add_argument("--aefc_depth", type=int, default=2)
    ap.add_argument("--aefc_dropout", type=float, default=0.1)
    ap.add_argument("--aefc_act", type=str, default="gelu", choices=["relu","silu","gelu"])
    ap.add_argument("--no_layernorm", action="store_false", dest="aefc_layernorm")
    ap.add_argument("--aefc_alpha", type=float, default=1.0)
    ap.add_argument("--mix_lambda", type=float, default=0.5)
    ap.add_argument("--x_bias", type=float, default=0.05, help="0 disables x-bias fallback; try 0.02~0.1")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}+cu{torch.version.cuda} | CUDA={torch.cuda.is_available()} | device={device}")

    ldm = build_model(args.ckpt)

    ref_wav, _ = librosa.load(args.ref, sr=args.sr)
    ref_wave = torch.tensor(ref_wav, dtype=torch.float32, device=device).unsqueeze(0)
    ref_len_s = max(0.25, len(ref_wav) / float(args.sr))
    eff_duration = min(args.duration, ref_len_s)

    to_mel = WaveToMel(sr=args.sr, hop=args.hop, n_mels=args.n_mels).to(device)
    obj = band_energy_objective(to_mel, band=(args.band_lo, args.band_hi))

    cond0 = to_mel(ref_wave).detach()  # [B,1,F,T]
    print(f"[cond0] {tuple(cond0.shape)} (mel latent)")

    # AEFC on last time dim
    D = int(cond0.shape[-1])
    aefc = DetachedAEFC(
        in_dim=D, hidden=args.aefc_hidden, depth=args.aefc_depth,
        dropout=args.aefc_dropout, activation=args.aefc_act,
        layernorm=args.aefc_layernorm, alpha=args.aefc_alpha
    ).to(device).eval()

    # ---------- Baselines ----------
    normal_out = force_ctx_generate(
        ldm, cond0, steps=args.inner_steps, guidance_scale=args.guidance,
        duration_s=eff_duration, seed=args.seed, ref_path=args.ref, sr=args.sr, x_bias=args.x_bias
    )
    cond0_aefc = aefc(cond0, ref=cond0, alpha=args.aefc_alpha)
    normal_out_aefc = force_ctx_generate(
        ldm, cond0_aefc, steps=args.inner_steps, guidance_scale=args.guidance,
        duration_s=eff_duration, seed=args.seed, ref_path=args.ref, sr=args.sr, x_bias=args.x_bias
    )

    # Random-condition probe
    e_rand = torch.randn_like(cond0) * (cond0.std().clamp_min(1e-6))
    random_out = force_ctx_generate(
        ldm, e_rand, steps=args.inner_steps, guidance_scale=args.guidance,
        duration_s=eff_duration, seed=args.seed, ref_path=args.ref, sr=args.sr, x_bias=args.x_bias
    )

    # ---------- Ascent ----------
    e = torch.nn.Parameter(cond0.clone().detach().requires_grad_(True))
    opt = torch.optim.Adam([e], lr=args.lr)

    for _ in tqdm(range(args.ascent_steps)):
        opt.zero_grad(set_to_none=True)
        e_aefc  = aefc(e, ref=cond0, alpha=args.aefc_alpha)
        e_used  = (1.0 - args.mix_lambda) * e + args.mix_lambda * e_aefc

        syn = force_ctx_generate(
            ldm, e_used, steps=args.inner_steps, guidance_scale=args.guidance,
            duration_s=eff_duration, seed=args.seed, ref_path=args.ref, sr=args.sr, x_bias=args.x_bias
        )
        score = obj(syn if syn.ndim > 1 else syn.unsqueeze(0))
        if score.ndim: score = score.mean()

        reg = 1e-3 * (e - cond0).pow(2).mean()
        loss = -(score) + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_([e], 5.0)
        opt.step()

        with torch.no_grad():
            d = e - cond0
            n = d.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
            e.data = cond0 + d * (args.tau / n).clamp(max=1.0)

    e_final = (1.0 - args.mix_lambda) * e + args.mix_lambda * aefc(e, ref=cond0, alpha=args.aefc_alpha)
    ascent_out = force_ctx_generate(
        ldm, e_final, steps=args.inner_steps, guidance_scale=args.guidance,
        duration_s=eff_duration, seed=args.seed, ref_path=args.ref, sr=args.sr, x_bias=args.x_bias
    )

    # ---------- Visuals ----------
    ref_wave_norm = normalize_rms(ref_wave, target_dbfs=args.norm_dbfs)
    mel_ref      = ensure_mel_b1ft(to_mel(ref_wave_norm))
    mel_normal   = to_mel_or_passthrough(normal_out, to_mel)
    mel_normal2  = to_mel_or_passthrough(normal_out_aefc, to_mel)
    mel_ascent   = to_mel_or_passthrough(ascent_out, to_mel)
    mel_random   = to_mel_or_passthrough(random_out, to_mel)

    def vmm(x):
        a = x[0,0].cpu().numpy().T
        return np.percentile(a, [2,98]).tolist()

    vmin_r, vmax_r   = vmm(mel_ref)
    vmin_n, vmax_n   = vmm(mel_normal)
    vmin_n2, vmax_n2 = vmm(mel_normal2)
    vmin_a, vmax_a   = vmm(mel_ascent)
    vmin_rd, vmax_rd = vmm(mel_random)

    plot_mel_time_y(mel_ref,     path="orig_norm.png",           sr=args.sr, hop=args.hop, title="Original (RMS-norm)",           vmin=vmin_r,  vmax=vmax_r)
    plot_mel_time_y(mel_normal,  path="normal_output.png",       sr=args.sr, hop=args.hop, title="Baseline (forced audio cond)",  vmin=vmin_n,  vmax=vmax_n)
    plot_mel_time_y(mel_normal2, path="normal_output_aefc.png",  sr=args.sr, hop=args.hop, title="Baseline + DetachedAEFC",       vmin=vmin_n2, vmax=vmax_n2)
    plot_mel_time_y(mel_ascent,  path="ascent_output.png",       sr=args.sr, hop=args.hop, title="Ascent + DetachedAEFC (mixed)", vmin=vmin_a,  vmax=vmax_a)
    plot_mel_time_y(mel_random,  path="random_output.png",       sr=args.sr, hop=args.hop, title="Random audio cond (probe)",     vmin=vmin_rd, vmax=vmax_rd)

    # ---------- Numeric diffs ----------
    def mse(a, b):
        A = to_mel_or_passthrough(a, to_mel)[0,0].cpu().numpy()
        B = to_mel_or_passthrough(b, to_mel)[0,0].cpu().numpy()
        Fm, Tm = min(A.shape[0], B.shape[0]), min(A.shape[1], B.shape[1])
        return float(np.mean((A[:Fm,:Tm] - B[:Fm,:Tm])**2))

    print("[mse] baseline vs baseline+AEFC:", f"{mse(normal_out, normal_out_aefc):.6e}")
    print("[mse] baseline vs ascent:",        f"{mse(normal_out, ascent_out):.6e}")
    print("[mse] baseline vs random:",        f"{mse(normal_out, random_out):.6e}")

    torch.save({
        "cond0": cond0.detach().cpu(),
        "e_final": e.detach().cpu(),
        "e_final_mixed": e_final.detach().cpu(),
        "mel_ref": mel_ref.cpu(),
        "mel_normal": mel_normal.cpu(),
        "mel_normal_aefc": mel_normal2.cpu(),
        "mel_ascent": mel_ascent.cpu(),
        "mel_random": mel_random.cpu(),
    }, "forced_ctx_outputs.pt")

    print("Wrote: orig_norm.png, normal_output.png, normal_output_aefc.png, ascent_output.png, random_output.png, forced_ctx_outputs.pt")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
