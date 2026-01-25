#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LibriSpeech activation extractor (Block OUTPUT):
- Walk LIBRI_ROOT for *.flac
- Read ref_text from CAPTION_LST (cross-sentence .lst)
- Run F5-TTS forward with hooks that capture each DiT block's OUTPUT (after residuals)
- Save per-ID activations as OUT_DIR/<ID>/<ID>_layer_{1..depth}.npy with shape (steps, H)
- Optionally dump generated wavs alongside OUT_DIR
"""

import os, re, json, shlex
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

# ====== path/setting ======
LIBRI_ROOT   = Path("<PATH for LibriSpeech unseen audio dir>")
CAPTION_LST  = Path("<PATH for librispeech_pc_test_clean_cross_sentence.lst>")

OUT_DIR      = Path("<PATH for activation dir for libri unseen>")

CONFIG_TOML  = Path("<PATH for infer/examples/basic/basic.toml>")
MODEL_NAME   = "F5TTS_v1_Base"
MODEL_CFG_YAML = Path("<PATH for configs/F5TTS_v1_Base.yaml>")
CKPT_FILE    = Path("<PATH for /ckpts/F5TTS_v1_Base/model_1250000.safetensors") # you can get ckpt through huggingface for F5-TTS

VOCAB_TXT    = ""
VOCODER_NAME = "vocos"   # "vocos" | "bigvgan"
LOAD_VOCODER_FROM_LOCAL = False

NFE_STEP       = 32
CFG_STRENGTH   = 2.0
SWAY_COEF      = -1.0
SPEED          = 1.0
FIX_DURATION   = None
TARGET_RMS     = -20.0
CROSS_FADE     = 0.15
DEVICE         = "cuda:1"

CAPTURE_COND_ONLY = True   # Collect only CFG cond paths
REF_LEN_FRAMES    = None   # If None, use full length L
SKIP_IF_DONE      = False  # Skip if you already have it

# saving options
SAVE_GEN_WAV   = True
GEN_OUT_DIR    = OUT_DIR
OVERWRITE_GEN  = True

# =====  =====
from trus.infer.utils_infer import (
    load_model, load_vocoder, preprocess_ref_audio_text,
    infer_process
)

# ========= caption loader =========
ID_RX = re.compile(r'([0-9]{3,5}-[0-9]{3,6}-[0-9]{4})')  # e.g., 908-157963-0000

def _clean_text(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1]
    return s.strip()

def _id_from(s: str) -> str:
    m = ID_RX.search(s)
    if m:
        return m.group(1)
    p = Path(s)
    if p.suffix.lower() == ".flac":
        return p.stem
    return s.strip()

def build_caption_map(lst_path: Path) -> dict:
    mp = {}
    if not lst_path.exists():
        print(f"[WARN] caption list not found: {lst_path}")
        return mp
    with lst_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if "\t" in line:
                cols = [c.strip() for c in line.split("\t")]
                if len(cols) >= 6:
                    id1 = _id_from(cols[0]); txt1 = _clean_text(cols[2])
                    id2 = _id_from(cols[3]); txt2 = _clean_text(cols[5])
                    if id1 and txt1: mp[id1] = txt1
                    if id2 and txt2: mp[id2] = txt2
                    continue
                elif len(cols) >= 3:
                    sid = _id_from(cols[0]); txt = _clean_text(cols[2])
                    if sid and txt: mp[sid] = txt
                    continue
            if "|" in line:
                left, text = line.split("|", 1)
                sid = _id_from(left); text = _clean_text(text)
                if sid and text: mp[sid] = text
                continue
            toks = shlex.split(line)
            if len(toks) >= 2:
                sid = _id_from(toks[0])
                text = _clean_text(line[len(toks[0]):])
                if sid and text: mp[sid] = text
                continue
    print(f"[INFO] captions loaded: {len(mp):,}")
    return mp

# ========= model/hook =========
def find_dit_and_blocks(model: nn.Module):
    for m in model.modules():
        if hasattr(m, "transformer_blocks") and isinstance(m.transformer_blocks, nn.ModuleList):
            return m, list(m.transformer_blocks)
    raise RuntimeError("DiT (with .transformer_blocks) not found.")

class ActTape:
    """
    Store (steps, H) by layer. Collect from block 'out'.
    """
    def __init__(self, steps=32, ref_len=None, capture_cond_only=True, num_layers=0):
        self.steps = steps
        self.ref_len = ref_len
        self.capture_cond_only = capture_cond_only
        self.current_step = 0
        self.store = [defaultdict(list) for _ in range(num_layers)]
        self._call_idx = 0

    def add(self, layer_idx: int, step: int, v: torch.Tensor):
        self.store[layer_idx][step].append(v.detach().to("cpu", dtype=torch.float32))

    def layer_stepsH(self):
        out = []
        for layer_dict in self.store:
            H = None
            for vecs in layer_dict.values():
                if vecs:
                    H = vecs[0].numel()
                    break
            if H is None:
                out.append(torch.empty((self.steps, 0)))
                continue
            rows = []
            for s in range(self.steps):
                vecs = layer_dict.get(s, [])
                rows.append(torch.zeros(H) if not vecs else torch.stack(vecs, 0).mean(dim=0))
            out.append(torch.stack(rows, 0))
        return out

def _try_scalar(x):
    try:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "flatten"):
            x = x.flatten()[0]
        return float(x)
    except Exception:
        return None

def make_time_tap(ctx: ActTape, steps=32):
    """
    Read time(0~1) of DiT.forward to trace step (fallback: call index)
    """
    def hook(module, input_args):
        step_from_time = None
        if input_args:
            for idx in (3, 2):
                if len(input_args) > idx:
                    val = _try_scalar(input_args[idx])
                    if val is not None and 0.0 <= val <= 1.0:
                        step_from_time = int(val * steps)
                        break
            if step_from_time is None:
                for obj in input_args:
                    val = _try_scalar(obj)
                    if val is not None and 0.0 <= val <= 1.0:
                        step_from_time = int(val * steps)
                        break
        step = step_from_time if step_from_time is not None else ctx._call_idx
        step = max(0, min(steps - 1, step))
        ctx.current_step = step
        ctx._call_idx = min(steps - 1, ctx._call_idx + 1)
        return
    return hook

def make_block_output_hook(ctx: ActTape, layer_idx: int):
    
    def hook(block, input_args, out):
        if not torch.is_tensor(out) or out.dim() != 3:
            return out
        y = out
        B, L, H = y.shape
        y_used = y[: B // 2] if (ctx.capture_cond_only and B % 2 == 0) else y
        refL = min(ctx.ref_len or L, L)
        eps = 1e-8
        y_slice = y_used[:, :refL, :]
        y_unit  = y_slice / (y_slice.norm(p=2, dim=-1, keepdim=True) + eps)
        v = y_unit.mean(dim=(0, 1))  # (H,)
        ctx.add(layer_idx, ctx.current_step, v)
        return out
    return hook

def register_capture_all_layers(model: nn.Module, steps=32, ref_len=None, capture_cond_only=True):
    dit, blocks = find_dit_and_blocks(model)
    tape = ActTape(steps=steps, ref_len=ref_len, capture_cond_only=capture_cond_only, num_layers=len(blocks))
    handles = []
    
    handles.append(dit.register_forward_pre_hook(make_time_tap(tape, steps=steps)))

    for idx, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_block_output_hook(tape, idx)))
    return tape, handles, dit, len(blocks)

def remove_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

# ========= model load =========
def load_everything():
    _ = tomli.load(open(CONFIG_TOML, "rb")) if CONFIG_TOML.exists() else {}
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
    model_cfg = OmegaConf.load(str(MODEL_CFG_YAML))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    ckpt_file = str(CKPT_FILE) if CKPT_FILE else str(cached_path(f"hf://SWivid/F5-TTS/{MODEL_NAME}/model_1250000.safetensors"))
    ema_model = load_model(model_cls, model_arc, ckpt_file, mel_spec_type=VOCODER_NAME, vocab_file=VOCAB_TXT, device=DEVICE)
    return ema_model, vocoder

# ========= save =========
def save_layerwise_stepsH(acts_per_layer, out_dir: Path, sample_id: str):
    dst = Path(out_dir) / sample_id
    dst.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(acts_per_layer, start=1):
        np.save(dst / f"{sample_id}_layer_{i}.npy", t.cpu().numpy())

# ========= sample =========
def run_one(ema_model, vocoder, ref_audio_path: Path, ref_text: str, sample_id: str):
    ref_audio_proc, ref_text_proc = preprocess_ref_audio_text(str(ref_audio_path), ref_text)

    tape, handles, dit_module, depth = register_capture_all_layers(
        ema_model, steps=NFE_STEP, ref_len=REF_LEN_FRAMES, capture_cond_only=CAPTURE_COND_ONLY
    )

    # checkpoint_activations  
    orig_ckpt_flag = bool(getattr(dit_module, "checkpoint_activations", False))
    if hasattr(dit_module, "checkpoint_activations"):
        dit_module.checkpoint_activations = False

    with torch.inference_mode():
        audio, sr, _ = infer_process(
            ref_audio_proc, ref_text_proc, ref_text_proc, ema_model, vocoder,
            mel_spec_type=VOCODER_NAME, target_rms=TARGET_RMS, cross_fade_duration=CROSS_FADE,
            nfe_step=NFE_STEP, cfg_strength=CFG_STRENGTH, sway_sampling_coef=SWAY_COEF,
            speed=SPEED, fix_duration=FIX_DURATION, device=DEVICE,
        )

    # wav saving(option)
    if SAVE_GEN_WAV:
        GEN_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_wav = GEN_OUT_DIR / f"{sample_id}_gen.wav"
        if OVERWRITE_GEN or not out_wav.exists():
            sf.write(str(out_wav), audio, sr, subtype="PCM_16")

    remove_hooks(handles)

    # Create and save a tensor for each layer (steps, H)
    acts_per_layer = tape.layer_stepsH()        # list[(steps,H)]
    save_layerwise_stepsH(acts_per_layer, OUT_DIR, sample_id)

    if hasattr(dit_module, "checkpoint_activations"):
        dit_module.checkpoint_activations = orig_ckpt_flag

# ========= main =========
def main():
    torch.set_grad_enabled(False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ema_model, vocoder = load_everything()
    cap_map = build_caption_map(CAPTION_LST)

    flacs = sorted(LIBRI_ROOT.rglob("*.flac"))
    print(f"[INFO] Found {len(flacs):,} flac files under {LIBRI_ROOT}")

    for flac in tqdm(flacs, desc="[ACT] libri", unit="file"):
        stem = flac.stem  # e.g., 908-157963-0000
        id_dir = OUT_DIR / stem
        if SKIP_IF_DONE and id_dir.exists() and any(id_dir.glob(f"{stem}_layer_*.npy")):
            continue

        ref_text = cap_map.get(stem, "")
        if not ref_text:
            print(f"[WARN] caption not found for {stem}, skip.")
            continue

        try:
            run_one(ema_model, vocoder, flac, ref_text, stem)
        except Exception as e:
            print(f"[ERR] {stem}: {e}")

    print("[DONE] saved wavs to:", GEN_OUT_DIR)
    print("[DONE] saved activations under per-ID dirs in:", OUT_DIR)

if __name__ == "__main__":
    main()
