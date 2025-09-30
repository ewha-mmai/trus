# for SIM-Emo

#!/usr/bin/env python3
import os
import json
import glob
import argparse
import re
import unicodedata
from collections import defaultdict
from typing import Optional, Any
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from transformers import AutoFeatureExtractor, AutoModel, Wav2Vec2FeatureExtractor
try:
    from funasr import AutoModel as FunASRModel
except Exception:
    FunASRModel = None

ALLOWED_EXTS = (".wav", ".mp3", ".flac")

def _id_parts(utt_id: str):
    parts = utt_id.split("_")
    sid = "_".join(parts[:-1]) if len(parts) >= 2 else utt_id
    wid = parts[-1] if parts else utt_id
    m = re.search(r"(\d+)$", wid)
    num = m.group(1) if m else wid
    return {"id": utt_id, "sid": sid, "wid": wid, "num": num}

def find_gen_audio(gen_root: str, utt_id: str, tpl: str, aggressive: bool=False):
    gen_root = gen_root.strip().rstrip("/ ")
    parts = _id_parts(utt_id.strip())

    try:
        candidate = os.path.join(gen_root, tpl.format(**parts))
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass

    stems = [parts["id"], f"{parts['sid']}_{parts['wid']}", parts["wid"], parts["num"]]
    for stem in stems:
        for ext in ALLOWED_EXTS:
            hit = os.path.join(gen_root, f"{stem}{ext}")
            if os.path.exists(hit):
                return hit

    if aggressive:
        for stem in stems:
            for ext in ALLOWED_EXTS:
                hits = glob.glob(os.path.join(gen_root, "**", f"{stem}{ext}"), recursive=True)
                if hits: return hits[0]
                hits = glob.glob(os.path.join(gen_root, "**", f"{stem}_*{ext}"), recursive=True)
                if hits: return hits[0]

        for ext in ALLOWED_EXTS:
            hits = glob.glob(os.path.join(gen_root, "**", f"*{parts['id']}*{ext}"), recursive=True)
            if hits: return hits[0]
            hits = glob.glob(os.path.join(gen_root, "**", f"*{parts['num']}*{ext}"), recursive=True)
            if hits: return hits[0]

    return None

def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            items.append(json.loads(line))
    return items

def _maybe_load_path(p: Any):
    if isinstance(p, str):
        if p.endswith(".npy") and os.path.exists(p):
            return np.load(p)
        if p.endswith(".txt") and os.path.exists(p):
            return np.loadtxt(p)
    return None

def _find_embedding_in_obj(obj: Any, debug: bool=False):
    candidate_keys = [
        "embedding", "embeddings", "emo_embedding",
        "spk_embedding", "speaker_embedding",
        "representation", "representations",
        "feat", "feats", "hidden_state", "hidden_states",
        "last_hidden_state"
    ]


    if isinstance(obj, dict):
        for k in candidate_keys:
            if k in obj and obj[k] is not None:
                arr = _maybe_load_path(obj[k])
                return arr if arr is not None else obj[k]

        for wrap in ["output", "outputs", "result", "results", "Hypotheses", "hypotheses"]:
            if wrap in obj:
                v = obj[wrap]
                got = _find_embedding_in_obj(v, debug)
                if got is not None:
                    return got

        for v in obj.values():
            got = _find_embedding_in_obj(v, debug)
            if got is not None:
                return got
        return None

    if isinstance(obj, (list, tuple)):
        for it in obj:
            got = _find_embedding_in_obj(it, debug)
            if got is not None:
                return got
        return None

    arr = _maybe_load_path(obj)
    if arr is not None:
        return arr

    try:
        if isinstance(obj, (np.ndarray,)):
            return obj
    except Exception:
        pass
    if torch.is_tensor(obj):
        return obj.cpu().numpy()

    return None


@torch.no_grad()
def extract_embedding(
    wav_path: str,
    feature_extractor,
    model,
    device: str,
    layer_index: int = -2,
    pool: str = "mean",      # "mean" or "cls"
    target_sr: Optional[int] = None,
    use_funasr: bool = False,
    funasr_model: Optional[object] = None,
    debug_funasr: bool = False,
):
    if use_funasr and funasr_model is not None:
        res = funasr_model.generate(
            wav_path,
            output_dir=None,
            granularity="utterance",
            extract_embedding=True,
        )
        if debug_funasr:
            print("[FUNASR-RAW]", type(res), res if isinstance(res, dict) else (res[:1] if isinstance(res, list) else res))

        emb_np = _find_embedding_in_obj(res, debug_funasr)
        if emb_np is None:
            raise RuntimeError("FunASR did not return an embedding-like field. Enable --debug_funasr to inspect.")

        if isinstance(emb_np, (list, tuple)):
            emb_np = torch.tensor(emb_np, dtype=torch.float32).numpy()
        if not isinstance(emb_np, np.ndarray):
            try:
                emb_np = np.array(emb_np, dtype="float32")
            except Exception:
                raise RuntimeError("Parsed object is not array-like. Enable --debug_funasr to inspect.")
        emb = torch.tensor(emb_np, dtype=torch.float32)
        if emb.dim() == 2 and emb.size(0) == 1:
            emb = emb.squeeze(0)
        emb = F.normalize(emb, dim=-1)
        return emb.detach().cpu()


    wav, sr = torchaudio.load(wav_path)

    if wav.dim() == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if target_sr and sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    wav_np = wav.squeeze(0).cpu().numpy()

    inputs = feature_extractor([wav_np], sampling_rate=sr, return_tensors="pt", padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs, output_hidden_states=True)
    hs = outputs.hidden_states[layer_index]

    if pool == "mean":
        emb = hs.mean(dim=1)
    elif pool == "cls":
        emb = hs[:, 0, :]
    else:
        raise ValueError("pool must be 'mean' or 'cls'")

    emb = F.normalize(emb, dim=-1)
    return emb.squeeze(0).detach().cpu()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_path", required=True)
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--gen_wav_dir", required=True)
    ap.add_argument("--model", default="emotion2vec/emotion2vec_plus_large")
    ap.add_argument("--gen_name_tpl", default="{id}.wav")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_missing_gen", action="store_true")
    ap.add_argument("--pool", choices=["mean","cls"], default="mean")
    ap.add_argument("--layer_index", type=int, default=-2)
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--debug_funasr", action="store_true", help="print raw FunASR outputs for the first few items")
    ap.add_argument("--aggressive_search", action="store_true", help="Recursive search is performed only when template matching fails.")
    args = ap.parse_args()

    device = args.device if (torch.cuda.is_available() and "cuda" in args.device) else "cpu"

    print(f"Loading model: {args.model}")_

    use_funasr = args.model.startswith("emotion2vec/emotion2vec_plus")
    funasr_model = None

    if use_funasr:
        if FunASRModel is None:
            raise RuntimeError("funasr is not installed. Install it or use a non-FunASR model.")

        ms_id = "iic/" + args.model.split("/")[-1]
        print(f"[INFO] Using FunASR pipeline: {ms_id}")
        funasr_model = FunASRModel(model=ms_id)
        feature_extractor = None
        model = None
        default_sr = 16000
        target_sr = args.target_sr or default_sr

    else:
        try:
            feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
        except Exception as e:
            print(f"[WARN] AutoFeatureExtractor load failed for {args.model}: {e}")
            print("[INFO] Falling back to a generic Wav2Vec2FeatureExtractor (16kHz, normalize, attention_mask).")
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=16000,
                padding_value=0.0,
                do_normalize=True,
                return_attention_mask=True,
            )

        model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(device).eval()
        
        default_sr = getattr(feature_extractor, "sampling_rate", None) or 16000
        target_sr = args.target_sr if (args.target_sr is not None) else int(default_sr)

        try:
            print("[DBG] FE sampling_rate:", getattr(feature_extractor, "sampling_rate", None))
            if hasattr(model, "config"):
                print("[DBG] model name_or_path:", getattr(model.config, "_name_or_path", None))
                print("[DBG] model_type:", getattr(model.config, "model_type", None))
        except Exception:
            pass

    items = load_jsonl(args.jsonl_path)
    os.makedirs(args.gen_wav_dir, exist_ok=True)
    out_path = os.path.join(args.gen_wav_dir, f"_{args.model.split('/')[-1]}_emo_sim_results.jsonl")

    sims = []
    by_class = defaultdict(list)
    detailed = []
    missing = 0

    printed_funasr = 0

    for ex in items:
        uid = ex["id"]
        ref_rel = ex["wav"]
        gt_emotion = (ex.get("emotion") or "").upper()

        ref_path = os.path.join(args.audio_root, ref_rel)
        gen_path = find_gen_audio(args.gen_wav_dir, uid, args.gen_name_tpl, aggressive=args.aggressive_search)
        if gen_path is None:
            msg = f"[WARN] missing gen audio for id={uid}"
            if args.skip_missing_gen:
                print(msg + " -> skip"); missing += 1; continue
            else:
                raise FileNotFoundError(msg)

        try:
            ref_emb = extract_embedding(
                ref_path, feature_extractor, model, device,
                layer_index=args.layer_index, pool=args.pool, target_sr=target_sr,
                use_funasr=use_funasr, funasr_model=funasr_model,
                debug_funasr=(args.debug_funasr and printed_funasr < 2)
            )
            gen_emb = extract_embedding(
                gen_path, feature_extractor, model, device,
                layer_index=args.layer_index, pool=args.pool, target_sr=target_sr,
                use_funasr=use_funasr, funasr_model=funasr_model,
                debug_funasr=(args.debug_funasr and printed_funasr < 2)
            )
            if args.debug_funasr and use_funasr and printed_funasr < 2:
                printed_funasr += 1
            sim = float(F.cosine_similarity(ref_emb, gen_emb, dim=0).item())
        except Exception as e:
            print(f"[ERROR] id={uid} -> {e}")
            continue

        sims.append(sim)
        if gt_emotion:
            by_class[gt_emotion].append(sim)

        detailed.append({
            "id": uid,
            "ref_wav": os.path.relpath(ref_path, args.audio_root),
            "gen_wav": os.path.relpath(gen_path, args.gen_wav_dir),
            "emotion": gt_emotion,
            "similarity": sim,
            "layer_index": args.layer_index,
            "pool": args.pool,
            "model": args.model,
        })

    mean_sim = float(sum(sims)/len(sims)) if sims else 0.0
    per_class = {k: float(sum(v)/len(v)) if v else 0.0 for k, v in sorted(by_class.items())}

    with open(out_path, "w", encoding="utf-8") as f:
        for row in detailed:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write("\n")
        summary = {
            "num_samples": len(sims),
            "missing_gen": missing,
            "overall_mean_similarity": round(mean_sim, 6),
            "per_class_mean_similarity": {k: round(v,6) for k,v in per_class.items()},
            "pool": args.pool,
            "layer_index": args.layer_index,
            "model": args.model,
            "target_sr": target_sr,
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"\nTotal eval = {len(sims)} (missing gen: {missing})")
    print(f"Overall mean cosine similarity = {mean_sim:.4f}")
    if per_class:
        print("Per-class mean similarity:")
        for k, v in per_class.items():
            print(f"  {k}: {v:.4f}")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()
