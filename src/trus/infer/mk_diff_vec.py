#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import csv

# ========== modify here ==========
FORGET_ROOT     = Path("<Path for forget activation>")
REMAIN_MEAN_DIR = Path("<Path for remain_mean activation")
OUT_ROOT        = Path("<Path for output - diff activation")

NUM_LAYERS   = 22
START_LAYER  = 1
MEAN_PREFIX  = "remain_30_mean_layer_"   
MEAN_SUFFIX  = ".npy"                    
OVERWRITE    = True                     
SAVE_FLAT_UNIT = False                  
EPS = 1e-8
# ========================================

def l2_norm(x: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    return float(np.sqrt(np.sum(x * x)))

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def load_mean(remain_mean_dir: Path, layer_idx: int,
              mean_prefix: str = MEAN_PREFIX, mean_suffix: str = MEAN_SUFFIX) -> np.ndarray:
    f = remain_mean_dir / f"{mean_prefix}{layer_idx}{mean_suffix}"
    if not f.is_file():
        f = remain_mean_dir / f"{mean_prefix}{layer_idx:01d}{mean_suffix}"
    if not f.is_file():
        f = remain_mean_dir / f"{mean_prefix}{layer_idx:02d}{mean_suffix}"
    if not f.is_file():
        raise FileNotFoundError(f"Remain mean not found for layer {layer_idx}: {f}")
    return np.load(str(f)).astype(np.float32, copy=False)

def load_forget_layer(sample_dir: Path, sample_id: str, layer_idx: int) -> np.ndarray:
    # basic rule: {ID}_layer_{i}.npy (i는 1..NUM_LAYERS)
    f = sample_dir / f"{sample_id}_layer_{layer_idx}.npy"
    if not f.is_file():
        f = sample_dir / f"{sample_id}_layer_{layer_idx:02d}.npy"
    if not f.is_file():
        # extra pattern
        candidates = [
            sample_dir / f"layer_{layer_idx}.npy",
            sample_dir / f"layer_{layer_idx:02d}.npy",
            sample_dir / f"{layer_idx}.npy",
        ]
        for c in candidates:
            if c.is_file():
                f = c
                break
    if not f.is_file():
        raise FileNotFoundError(f"Forget layer not found: {sample_dir} (L{layer_idx})")
    return np.load(str(f)).squeeze().astype(np.float32, copy=False)

def _ensure_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x
    if x.ndim == 1:
        if x.shape[0] % 32 == 0:
            return x.reshape(32, -1)
        else:
            return x.reshape(1, -1)
    x = x.reshape(-1)
    if x.shape[0] % 32 == 0:
        return x.reshape(32, -1)
    else:
        return x.reshape(1, -1)

def process_one_sample(sample_dir: Path, remain_mean_dir: Path, out_dir: Path,
                       num_layers: int, start_layer: int = 1):
    sample_id = sample_dir.name  # e.g. EN_B00000_S00290_W000001
    out_dir.mkdir(parents=True, exist_ok=True)

    if not OVERWRITE:
        done = all((out_dir / f"{sample_id}_layer_{li}_diff_unit2d.npy").exists()
                   for li in range(start_layer, start_layer + num_layers))
        if done and (out_dir / "diff_summary.csv").exists():
            return

    diff_list = []
    summary_rows = [["layer_idx", "vec_dim", "forget_F", "mean_F", "diff_F", "cos(forget,mean)"]]

    for li in range(start_layer, start_layer + num_layers):
        try:
            fvec = load_forget_layer(sample_dir, sample_id, li)
            mvec = load_mean(remain_mean_dir, li)
        except FileNotFoundError as e:
            print(f"[WARN] {sample_id} L{li}: {e}")
            diff_list.append(None)
            continue

        fvec = _ensure_2d(fvec)
        mvec = _ensure_2d(mvec)

        T = min(fvec.shape[0], mvec.shape[0])
        D = min(fvec.shape[1], mvec.shape[1])
        if (fvec.shape[0], fvec.shape[1]) != (T, D) or (mvec.shape[0], mvec.shape[1]) != (T, D):
            print(f"[WARN] {sample_id} L{li}: shape mismatch {fvec.shape} vs {mvec.shape} -> cut to ({T}, {D})")
        fvec = fvec[:T, :D]
        mvec = mvec[:T, :D]

        diff = (fvec - mvec).astype(np.float32)
        normF = float(np.linalg.norm(diff))  # = np.linalg.norm(diff.ravel())
        diff_unit2d = (diff / (normF + EPS)).astype(np.float32) if normF > 0 else np.zeros_like(diff, dtype=np.float32)

        # save
        np.save(out_dir / f"{sample_id}_layer_{li}_diff.npy", diff)                     # original 2D
        np.save(out_dir / f"{sample_id}_layer_{li}_diff_unit2d.npy", diff_unit2d)       # 2D + ||·||_F=1

        if SAVE_FLAT_UNIT:
            flat_unit = (diff.reshape(-1) / (normF + EPS)).astype(np.float32) if normF > 0 else np.zeros(diff.size, dtype=np.float32)
            np.save(out_dir / f"{sample_id}_layer_{li}_diff_unit.npy", flat_unit)

        print(f"[DEBUG] {sample_id} L{li}: diff {diff.shape}, unit2d {diff_unit2d.shape} | "
              f"||diff||F={normF:.6f}, ||unit2d||F={np.linalg.norm(diff_unit2d):.6f}")

        diff_list.append(diff)

        try:
            summary_rows.append([
                li, D,
                l2_norm(fvec), l2_norm(mvec), normF,
                cosine_sim(fvec.ravel(), mvec.ravel())
            ])
        except Exception as e:
            print(f"[WARN] {sample_id} L{li}: summary fail ({e})")

    kept = [d for d in diff_list if d is not None]
    if kept:
        dims = {d.shape for d in kept}
        if len(dims) == 1:
            stack = np.stack(kept, axis=0).astype(np.float32)  # [num_layers_kept, T, D]
            np.save(out_dir / "diff_stack.npy", stack)
            print(f"[INFO] Saved diff_stack.npy with shape {stack.shape}")
        else:
            np.save(out_dir / "diff_stack_object.npy", np.array(kept, dtype=object))
            print(f"[INFO] Saved diff_stack_object.npy (varying shapes)")

    with open(out_dir / "diff_summary.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(summary_rows)

def main():
    assert FORGET_ROOT.is_dir(), f"forget_root not found: {FORGET_ROOT}"
    assert REMAIN_MEAN_DIR.is_dir(), f"remain_mean_dir not found: {REMAIN_MEAN_DIR}"
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted([p for p in FORGET_ROOT.iterdir() if p.is_dir()])
    print(f"[INFO] # of found forget samples: {len(sample_dirs)}")

    for sd in tqdm(sample_dirs, desc="Processing forget samples"):
        out_dir = OUT_ROOT / sd.name
        process_one_sample(sd, REMAIN_MEAN_DIR, out_dir, num_layers=NUM_LAYERS, start_layer=START_LAYER)

    print("[DONE] Complete all diff calculation!!")

if __name__ == "__main__":
    main()
