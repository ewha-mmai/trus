# for spk-ZRF-F

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import glob
import re
import unicodedata
import gc
from typing import List, Dict, Any

import numpy as np
import math
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMForXVector

sys.path.insert(0, "F5-TTS/src")
sys.path.append(os.getcwd())


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl_path", type=str, required=True, help="path to emilia_forget_test.jsonl")
    p.add_argument("--audio_root", type=str, required=True, help="root to prepend to 'wav' (consistency/logging)")
    p.add_argument("-l", "--lang", type=str, default=None, help="optional language filter, e.g., 'en'")

    p.add_argument("-g", "--gen_wav_dir", type=str, required=True,
                   help="directory containing θ (text-only) generated audios")
    p.add_argument("--gen_wav_dir_minus", type=str, required=True,
                   help="directory containing θ⁻ (text + speaker prompt) generated audios")
    p.add_argument("--gen_name_tpl", type=str, default="{id}.wav",
                   help="filename template. Tokens: {id}, {sid}, {wid}, {num}")
    p.add_argument("--skip_missing_gen", action="store_true",
                   help="skip items when either θ or θ⁻ audio is missing instead of raising")

    p.add_argument("--min_dur", type=float, default=0.0, help="drop samples shorter than this (sec)")
    p.add_argument("--max_dur", type=float, default=0.0, help="drop samples longer than this (sec), 0=disabled")

    p.add_argument("--sv_model_name", type=str, default="microsoft/wavlm-base-plus-sv",
                   help="HuggingFace model id or local path")
    p.add_argument("--sv_target_sr", type=int, default=16000,
                   help="target sampling rate for SV (audio will be resampled)")
    p.add_argument("--alpha", type=float, default=5.5,
                   help="temperature for softmax over cosine logits")
    p.add_argument("--pooling", type=str, default="speaker",
                   choices=["utterance", "speaker"],
                   help="class axis for softmax: utterance-level or speaker-level pooling")

    p.add_argument("--batch_size", type=int, default=8, help="θ/θ⁻ forward batch size")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")

    p.add_argument("--embed_batch_size", type=int, default=2,
                   help="mini-batch size for embedding extraction (enroll/θ/θ⁻). Use 1~4 to avoid OOM")
    p.add_argument("--max_seconds", type=float, default=8.0,

                   help="trim/clamp each waveform to this length (seconds) before feature extraction")
    p.add_argument("--no_amp", action="store_true",
                   help="disable autocast (fp16) inference on CUDA")

    return p.parse_args()

ALLOWED_EXTS = (".wav", ".mp3", ".flac")

def _clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return s.strip().replace("\r", "").replace("\n", "").replace("\t", "")

def _id_parts(utt_id: str):
    parts = utt_id.split("_")
    sid = "_".join(parts[:-1]) if len(parts) >= 2 else utt_id
    wid = parts[-1] if parts else utt_id
    m = re.search(r"(\d+)$", wid)
    num = m.group(1) if m else wid
    return {"id": utt_id, "sid": sid, "wid": wid, "num": num}

def find_gen_audio(gen_root: str, utt_id: str, tpl: str):
    gen_root = _clean(gen_root).rstrip("/ ")
    parts = _id_parts(_clean(utt_id))

    try:
        candidate = os.path.join(gen_root, tpl.format(**parts))
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass

    stems = [parts["id"], parts["sid"] + "_" + parts["wid"], parts["wid"], parts["num"]]
    for stem in stems:
        for ext in ALLOWED_EXTS:
            hit = os.path.join(gen_root, f"{stem}{ext}")
            if os.path.exists(hit):
                return hit

    for stem in stems:
        for ext in ALLOWED_EXTS:
            hits = glob.glob(os.path.join(gen_root, "**", f"{stem}{ext}"), recursive=True)
            if hits:
                return hits[0]
            hits = glob.glob(os.path.join(gen_root, "**", f"{stem}_*{ext}"), recursive=True)
            if hits:
                return hits[0]

    for ext in ALLOWED_EXTS:
        hits = glob.glob(os.path.join(gen_root, "**", f"*{parts['id']}*{ext}"), recursive=True)
        if hits:
            return hits[0]
        hits = glob.glob(os.path.join(gen_root, "**", f"*{parts['num']}*{ext}"), recursive=True)
        if hits:
            return hits[0]
    return None

def read_emilia_jsonl(jsonl_path, audio_root, lang_filter=None, min_dur=0.0, max_dur=0.0):
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            ex = json.loads(line)

            if lang_filter and ex.get("language") != lang_filter:
                continue

            dur = float(ex.get("duration", 0.0) or 0.0)
            if min_dur and dur < min_dur:
                continue
            if max_dur and max_dur > 0 and dur > max_dur:
                continue

            ex_id = _clean(ex["id"])
            ex_wav = _clean(ex["wav"])
            text = _clean(ex.get("text", ""))

            ref_abs = os.path.join(_clean(audio_root), ex_wav)

            if not os.path.exists(ref_abs):
                base = os.path.basename(ex_wav)
                ref_abs2 = os.path.join(_clean(audio_root), base)
                if os.path.exists(ref_abs2):
                    ref_abs = ref_abs2
                else:
                    alt = os.path.join(_clean(audio_root), f"{ex_id}.mp3")
                    if os.path.exists(alt):
                        ref_abs = alt
                    else:
                        print(f"[WARN] ref audio not found: {ref_abs} or {ref_abs2} or {alt} -> skip '{ex_id}'")
                        continue

            items.append({
                "id": ex_id,
                "text": text,
                "ref_wav": ref_abs,
                "speaker": _clean(ex.get("speaker", "")),
                "language": _clean(ex.get("language", "")),
                "duration": dur,
                "dnsmos": ex.get("dnsmos", None),
            })
    return items


def build_testset_spkzrf(items, gen_root_theta, gen_root_tminus,
                         skip_missing_gen=False, gen_name_tpl="{id}.wav"):
    pairs = []
    miss = 0
    for ex in items:
        utt_id = _clean(ex["id"])
        p_theta = find_gen_audio(gen_root_theta, utt_id, gen_name_tpl)
        p_tminus = find_gen_audio(gen_root_tminus, utt_id, gen_name_tpl)
        if p_theta is None or p_tminus is None:
            miss += 1
            msg = (f"[WARN] Missing generated audio for id={utt_id} "
                   f"(theta={bool(p_theta)} tminus={bool(p_tminus)})")
            if skip_missing_gen:
                print(msg + " -> skip")
                continue
            else:
                print(f"[WARN] Generated wav not found: {utt_id}", flush=True)
                continue
        def _resolve_spk_id(ex):
            if ex.get("spk_id"):
                return ex["spk_id"]
            if ex.get("speaker"):
                return ex["speaker"]
            m = re.match(r"^(\d+)-\d+-\d+$", ex["id"])
            return m.group(1) if m else ex["id"]

        spk_id = _resolve_spk_id(ex)
        pairs.append({
            "id": utt_id,
            "theta": p_theta,
            "tminus": p_tminus,
            "enroll": ex["ref_wav"],
            "spk_id": spk_id
        })
    if miss:
        print(f"[INFO] Missing pairs for {miss} ids")
    return pairs

def load_wav(path, target_sr):
    wav, sr = torchaudio.load(path)
    if wav.dtype != torch.float32:
        wav = wav.to(torch.float32)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr
    return wav.squeeze(0), sr

def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (p.log() - m.log()), dim=-1)
    kl_qm = torch.sum(q * (q.log() - m.log()), dim=-1)
    return 0.5 * (kl_pm + kl_qm)

def run_spkzrf(
    pairs,
    sv_model_name,
    target_sr,
    batch_size=8,
    device=None,
    embed_batch_size=2,
    max_seconds=8.0,
    use_amp=True,
    alpha=5.5,
    pooling="speaker"
):

    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    feature_extractor = AutoFeatureExtractor.from_pretrained(sv_model_name)
    backbone = WavLMForXVector.from_pretrained(sv_model_name).to(device)
    backbone.eval()

    def _embed(wavs_list):
        sr = target_sr
        max_len = int(sr * float(max_seconds)) if max_seconds and max_seconds > 0 else None
        embs_cpu = []

        for i in range(0, len(wavs_list), embed_batch_size):
            chunk = wavs_list[i:i + embed_batch_size]
            proc_chunk = []
            for w in chunk:
                if isinstance(w, torch.Tensor):
                    w = w.detach().cpu().float().numpy()
                else:
                    w = np.asarray(w, dtype=np.float32)
                if max_len is not None and w.shape[-1] > max_len:
                    w = w[:max_len]
                proc_chunk.append(w)

            inputs = feature_extractor(
                proc_chunk,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len if max_len is not None else None,
            )
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            with torch.inference_mode():
                if device.type == "cuda" and use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        emb = backbone(**inputs).embeddings  # (b, D)
                else:
                    emb = backbone(**inputs).embeddings

            embs_cpu.append(emb.detach().cpu())

            del inputs, emb
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if embs_cpu:
            return torch.cat(embs_cpu, dim=0)  # (B, D)
        else:
            hidden = getattr(backbone.config, "hidden_size", None)
            if hidden is None:
                hidden = getattr(backbone.config, "projection_dim", 256)
            return torch.empty(0, hidden)

    def _zscore_rowwise(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mean) / (std + eps)

    def _js_divergence(P: torch.Tensor, Q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        P = P.clamp(min=eps)
        Q = Q.clamp(min=eps)
        M = 0.5 * (P + Q)
        kl_pm = torch.sum(P * (P.log() - M.log()), dim=-1)
        kl_qm = torch.sum(Q * (Q.log() - M.log()), dim=-1)
        jsd = 0.5 * (kl_pm + kl_qm)
        jsd = jsd / math.log(2.0)
        return jsd.clamp(min=0.0, max=1.0)

    from tqdm import tqdm
    import torchaudio

    def _load_wav(path, target_sr):
        wav, sr = torchaudio.load(path)  # (C, T)
        if wav.dtype != torch.float32:
            wav = wav.to(torch.float32)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)  # mono
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.squeeze(0)  # (T,)

    print("[spk-ZRF] mode = feature-axis distribution (dim = D); alpha/pooling ignored.")

    results, jsd_vals = [], []
    ids, buf_theta, buf_tminus = [], [], []

    def _process_batch(ids_buf, th_buf, tm_buf):
        if not th_buf:
            return
        T = _embed(th_buf)
        M = _embed(tm_buf)

        Tz = _zscore_rowwise(T)
        Mz = _zscore_rowwise(M)
        P = torch.softmax(Tz, dim=1)  # (B, D)
        Q = torch.softmax(Mz, dim=1)  # (B, D)

        jsd = _js_divergence(P, Q)    # (B,)

        for k in range(jsd.shape[0]):
            v = float(jsd[k].item())
            results.append({"id": ids_buf[k], "jsd": v, "spkzrf": float(1.0 - v)})
            jsd_vals.append(v)

        del T, M, Tz, Mz, P, Q, jsd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for item in tqdm(pairs, desc="Computing spk-ZRF (feature-axis)"):
        wav_t = _load_wav(item["theta"], target_sr)
        wav_m = _load_wav(item["tminus"], target_sr)
        ids.append(item["id"])
        buf_theta.append(wav_t.numpy())
        buf_tminus.append(wav_m.numpy())

        if len(buf_theta) >= batch_size:
            _process_batch(ids, buf_theta, buf_tminus)
            ids, buf_theta, buf_tminus = [], [], []

    _process_batch(ids, buf_theta, buf_tminus)

    spkzrf_mean = float(1.0 - np.mean(jsd_vals)) if jsd_vals else 0.0
    return results, spkzrf_mean


def main():
    args = get_args()

    items = read_emilia_jsonl(
        jsonl_path=args.jsonl_path,
        audio_root=args.audio_root,
        lang_filter=args.lang,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
    )

    pairs = build_testset_spkzrf(
        items,
        gen_root_theta=args.gen_wav_dir,
        gen_root_tminus=args.gen_wav_dir_minus,
        skip_missing_gen=args.skip_missing_gen,
        gen_name_tpl=args.gen_name_tpl,
    )
    print(f"[INFO] Paired {len(pairs)} ids for spk-ZRF")

    device = torch.device("cpu") if args.cpu else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    use_amp = (not args.no_amp)
    results, spkzrf_mean = run_spkzrf(
        pairs,
        sv_model_name=args.sv_model_name,
        target_sr=args.sv_target_sr,
        batch_size=args.batch_size,
        device=device,
        embed_batch_size=args.embed_batch_size,
        max_seconds=args.max_seconds,
        use_amp=use_amp,
        alpha=args.alpha,
        pooling=args.pooling
    )

    os.makedirs(args.gen_wav_dir, exist_ok=True)
    out_path = os.path.join(os.path.dirname(args.gen_wav_dir), "_spkzrf_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(f"\nSPKZRF: {round(spkzrf_mean, 5)}\n")

    print(f"\nTotal {len(results)} samples")
    print(f"SPKZRF: {spkzrf_mean:.5f}")
    print(f"SPKZRF results saved to {out_path}")

if __name__ == "__main__":
    main()
