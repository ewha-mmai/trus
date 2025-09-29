#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch activation extractor (Block OUTPUT, fixed gen text, no argparse):
- 각 DiT 블록의 '출력(out)' 지점에서 per-step (H,) 벡터를 캡처
- CFG cond-only 수집 옵션 제공
- 저장: OUT_DIR/<id>/<id>_layer_{1..depth}.npy  (각 파일 shape=(steps, H))
"""

import os, re, json, codecs, shutil
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm
from unidecode import unidecode

# ====== modify here ======
#AUDIO_DIR      = Path("<Path for data/Emilia_out/audio/forget>")
AUDIO_DIR      = Path("<Path for audio>")

META_DIR       = AUDIO_DIR
#OUT_DIR        = Path("<Path for /data/Emilia_out/forget")
OUT_DIR        = Path("<Path for output - activation")

CONFIG_TOML    = Path("<Path for /infer/examples/basic/basic.toml>")
MODEL_NAME     = "F5TTS_v1_Base"
MODEL_CFG_YAML = Path("<Path for configs/F5TTS_v1_Base.yaml>")
CKPT_FILE      = Path("<Path for ckpts/F5TTS_v1_Base/model_1250000.safetensors>")
VOCAB_TXT      = ""
VOCODER_NAME   = "vocos"  # "vocos" | "bigvgan"
LOAD_VOCODER_FROM_LOCAL = False

# ── Fixed generated text──────────────────────────────────────────────
GEN_TEXT       = "I loved you dangerously. More than the air that I breathe. Knew we would crash at the speed that we were going. Didn’t care if the explosion ruined me."
# ─────────────────────────────────────────────────────────────────

NFE_STEP       = 32
CFG_STRENGTH   = 2.0
SWAY_COEF      = -1.0
SPEED          = 1.0
FIX_DURATION   = None
TARGET_RMS     = -20.0
CROSS_FADE     = 0.15
DEVICE         = "cuda:0"

CAPTURE_COND_ONLY = True   # recommend to align both extraction and injection based on the cond path.
REF_LEN_FRAMES    = None   # If None, the entire L
SKIP_IF_DONE      = True

# save option
SAVE_GEN_WAV   = True
GEN_OUT_DIR    = OUT_DIR
OVERWRITE_GEN  = True
# ===================================

# ===== existing util =====
from trus.infer.utils_infer import (
    load_model, load_vocoder, preprocess_ref_audio_text,
    mel_spec_type as _mel_spec_type_default,
    infer_process
)

# ---------- Model navigation/hook/save utility ----------
def find_dit_and_blocks(model: nn.Module):
    for m in model.modules():
        if hasattr(m, "transformer_blocks") and isinstance(m.transformer_blocks, nn.ModuleList):
            return m, list(m.transformer_blocks)
    raise RuntimeError("DiT (with .transformer_blocks) not found.")

class ActTape:
    def __init__(self, steps=32, ref_len=None, capture_cond_only=True):
        self.steps = steps
        self.ref_len = ref_len
        self.capture_cond_only = capture_cond_only
        self.current_step = 0
        self.store = defaultdict(dict)  # {slot: {step: (H,)}}
        self._call_idx = 0

    def put(self, slot, step, v: torch.Tensor):
        v = v.detach().to("cpu", dtype=torch.float32)
        prev = self.store[slot].get(step)
        self.store[slot][step] = v if prev is None else 0.5 * (prev + v)

    def as_tensor(self, device="cpu"):
        slots = sorted(self.store.keys())
        if not slots:
            return torch.empty(0, device=device)
        steps = self.steps
        H = None
        for s in slots:
            for vv in self.store[s].values():
                H = vv.numel()
                break
            if H is not None:
                break
        if H is None:
            return torch.empty((len(slots), steps, 0), device=device)

        out = []
        z = torch.zeros(H)
        for s in slots:
            row = []
            for k in range(steps):
                row.append(self.store[s].get(k, z))
            out.append(torch.stack(row, 0))
        return torch.stack(out, 0).to(device)  # [num_layers, steps, H]

def make_time_tap(ctx: ActTape, steps=32):
    def hook(module, input_args):
        step_from_time = None
        if input_args and len(input_args) >= 4:
            try:
                t = input_args[3]
                t0 = float(t.flatten()[0].item()) if hasattr(t, "flatten") else float(t)
                if 0.0 <= t0 <= 1.0:
                    step_from_time = int(t0 * steps)
            except Exception:
                pass
        if step_from_time is None:
            step = min(steps - 1, ctx._call_idx)
        else:
            step = max(0, min(steps - 1, step_from_time))
        ctx.current_step = step
        ctx._call_idx = min(steps - 1, ctx._call_idx + 1)
        return
    return hook

def make_block_output_hook(ctx: ActTape, slot: int):

    def hook(block, inp, out):
        if not torch.is_tensor(out):
            return out
        y = out
        if y.dim() != 3:
            return out
        B, L, H = y.shape
        y_used = y[: B // 2] if (ctx.capture_cond_only and B % 2 == 0) else y
        refL = min(ctx.ref_len or L, L)
        eps = 1e-8
        y_slice = y_used[:, :refL, :]
        y_unit  = y_slice / (y_slice.norm(p=2, dim=-1, keepdim=True) + eps)  # per-token normalize
        v = y_unit.mean(dim=(0, 1))  # (H,)
        ctx.put(slot, ctx.current_step, v)
        return out
    return hook

def register_capture_all_layers(model: nn.Module, steps=32, ref_len=None, capture_cond_only=True):
    dit, blocks = find_dit_and_blocks(model)
    tape = ActTape(steps=steps, ref_len=ref_len, capture_cond_only=capture_cond_only)
    handles = []
    handles.append(dit.register_forward_pre_hook(make_time_tap(tape, steps=steps)))
    for slot, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_block_output_hook(tape, slot)))
    return tape, handles, len(blocks), dit

def remove_hooks(handles):
    for h in handles:
        try: h.remove()
        except: pass

def save_layerwise(acts_tensor: torch.Tensor, out_dir: Path, sample_id: str):
    out_dir = Path(out_dir) / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    acts_np = acts_tensor.cpu().numpy()
    num_layers = acts_np.shape[0]
    for k in range(num_layers):
        np.save(out_dir / f"{sample_id}_layer_{k+1}.npy", acts_np[k])

# ---------- Search meta ----------
def read_text_from_meta(meta_dir: Path, stem: str):
    p1 = meta_dir / f"{stem}.json"
    if p1.exists():
        try:
            with p1.open("r", encoding="utf-8") as f:
                return json.load(f).get("text", None)
        except Exception:
            pass
    p2 = meta_dir / f"{stem}.jsonl"
    if p2.exists():
        try:
            with p2.open("r", encoding="utf-8") as f:
                for line in f:
                    line=line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("id","").startswith(stem) or obj.get("wav","").endswith(stem + ".mp3"):
                        return obj.get("text", None)
        except Exception:
            pass
    return None

# ---------- main ----------
def load_everything():
    _config = tomli.load(open(CONFIG_TOML, "rb")) if CONFIG_TOML.exists() else {}
    # vocoder
    if VOCODER_NAME == "vocos":
        vocoder_local_path = "../checkpoints/vocos-mel-24khz"
    elif VOCODER_NAME == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    else:
        vocoder_local_path = "../checkpoints/vocos-mel-24khz"
    vocoder = load_vocoder(
        vocoder_name=VOCODER_NAME,
        is_local=LOAD_VOCODER_FROM_LOCAL,
        local_path=vocoder_local_path,
        device=DEVICE
    )
    # model
    model_cfg = OmegaConf.load(str(MODEL_CFG_YAML))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    ckpt_file = str(CKPT_FILE) if CKPT_FILE else str(cached_path(f"hf://SWivid/F5-TTS/{MODEL_NAME}/model_1250000.safetensors"))
    ema_model = load_model(model_cls, model_arc, ckpt_file, mel_spec_type=VOCODER_NAME, vocab_file=VOCAB_TXT, device=DEVICE)
    return ema_model, vocoder, _config

def run_one(ema_model, vocoder, ref_audio_path: Path, ref_text: str, sample_id: str):
    ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(str(ref_audio_path), ref_text)

    tape, handles, depth, dit_module = register_capture_all_layers(
        ema_model,
        steps=NFE_STEP,
        ref_len=REF_LEN_FRAMES,
        capture_cond_only=CAPTURE_COND_ONLY
    )

    # (option) off gradient checkpointing 
    orig_ckpt_flag = bool(getattr(dit_module, "checkpoint_activations", False))
    if hasattr(dit_module, "checkpoint_activations"):
        dit_module.checkpoint_activations = False

    # forward
    _audio, _sr, _spec = infer_process(
        ref_audio_proc, ref_text_proc, GEN_TEXT, ema_model, vocoder,
        mel_spec_type=VOCODER_NAME, target_rms=TARGET_RMS, cross_fade_duration=CROSS_FADE,
        nfe_step=NFE_STEP, cfg_strength=CFG_STRENGTH, sway_sampling_coef=SWAY_COEF,
        speed=SPEED, fix_duration=FIX_DURATION, device=DEVICE,
    )

    # save gen wav 
    if SAVE_GEN_WAV:
        GEN_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_wav = GEN_OUT_DIR / f"{sample_id}_gen.wav"
        if OVERWRITE_GEN or not out_wav.exists():
            sf.write(str(out_wav), _audio, _sr)

    remove_hooks(handles)
    acts = tape.as_tensor(device="cpu")  # [num_layers, steps, H]
    save_layerwise(acts, OUT_DIR, sample_id)

    if hasattr(dit_module, "checkpoint_activations"):
        dit_module.checkpoint_activations = orig_ckpt_flag

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ema_model, vocoder, _cfg = load_everything()

    mp3s = sorted([p for p in AUDIO_DIR.glob("*.mp3")])
    print(f"[INFO] Found {len(mp3s)} mp3 files in {AUDIO_DIR}")

    for mp3 in tqdm(mp3s, desc="[ACT] extracting", unit="file"):
        stem = mp3.stem
        sample_out_dir = OUT_DIR / stem
        if SKIP_IF_DONE and sample_out_dir.exists() and any(sample_out_dir.glob(f"{stem}_layer_*.npy")):
            continue

        ref_text = read_text_from_meta(META_DIR, stem)
        if not ref_text:
            print(f"[WARN] text not found for {stem}, skipping.")
            continue

        try:
            run_one(ema_model, vocoder, mp3, ref_text, stem)
        except Exception as e:
            print(f"[ERR] failed on {stem}: {e}")

    print("[DONE] all activations saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
