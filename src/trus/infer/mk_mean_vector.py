# Collect npy for each speaker for layer k and average them
# save as mean_10_layer_k.npy (10 : # of remain speakers)

import os, json, glob
from pathlib import Path
import numpy as np

def list_speaker_dirs(root_dir: str):
    return sorted([p for p in Path(root_dir).iterdir() if p.is_dir()])

def discover_layers_from_one_speaker(speaker_dir: Path):
    files = sorted(speaker_dir.glob("*_layer_*.npy"))
    print(f"num of npy per speaker : {speaker_dir} =======> {len(files)}\n")
    if not files:
        raise RuntimeError(f"No layer files in {speaker_dir}")
    layer_map = {}
    for f in files:
        name = f.stem  # e.g., EN_B00000_S00290_W000001_layer_1
        try:
            k = int(name.split("_layer_")[-1])
        except Exception:
            raise RuntimeError(f"Cannot parse layer id from filename: {f.name}")
        layer_map[k] = f
    return layer_map

def load_layer_array(path: Path, want_dtype=np.float32):
    arr = np.load(path)
    if arr.dtype != want_dtype:
        arr = arr.astype(want_dtype)
    if not np.isfinite(arr).all():
        raise ValueError(f"Found NaN/Inf in {path}")
    return arr  # shape: (steps, hidden)

def make_layer_means(root_dir: str, out_subdir="_layer_means", verbose=True):
    root = Path(root_dir)
    speakers = list_speaker_dirs(root_dir)
    print(len(speakers))
    assert speakers, f"No speaker folders under {root_dir}"

    first_layer_map = discover_layers_from_one_speaker(speakers[0])
    layer_ids = sorted(first_layer_map.keys())
    if verbose:
        print(f"Found {len(layer_ids)} layers from {speakers[0].name}: {layer_ids[:8]}{'...' if len(layer_ids)>8 else ''}")

    sums = {}
    counts = {k: 0 for k in layer_ids}
    steps_hidden = None

    used_speakers = []
    for sp in speakers:
        layer_map = discover_layers_from_one_speaker(sp)
        miss = [k for k in layer_ids if k not in layer_map]
        if miss:
            if verbose:
                print(f"[WARN] Skip speaker {sp.name}: missing layers {miss[:5]}{'...' if len(miss)>5 else ''}")
            continue

        shapes_ok = True
        per_speaker_loaded = {}
        for k in layer_ids:
            arr = load_layer_array(layer_map[k])
            per_speaker_loaded[k] = arr
            if steps_hidden is None:
                steps_hidden = arr.shape
            elif arr.shape != steps_hidden:
                shapes_ok = False
                if verbose:
                    print(f"[WARN] shape mismatch at {sp.name} layer {k}: {arr.shape} != {steps_hidden}")
                break
        if not shapes_ok:
            continue

        for k in layer_ids:
            a = per_speaker_loaded[k].astype(np.float64)
            if k not in sums:
                sums[k] = a.copy()
            else:
                sums[k] += a
            counts[k] += 1
        used_speakers.append(sp.name)

    assert sums, "Nothing accumulated. Check folder structure / file names."

    #out_dir = base_dir / out_subdir
    ###################################### set save path here
    out_dir = Path("<path for remain_mean>")
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"root": str(root), "out_dir": str(out_dir), "steps_hidden": steps_hidden,
             "num_layers": len(layer_ids), "num_speakers_used": len(used_speakers),
             "speakers_used": used_speakers}

    for k in layer_ids:
        if counts[k] == 0:
            if verbose:
                print(f"[SKIP] layer {k}: no speakers")
            continue
        mean_k = (sums[k] / counts[k]).astype(np.float32)   # (steps, hidden)
        ############################## If the number of speakers changes here, the file name will also change.
        np.save(out_dir / f"remain_10_mean_layer_{k}.npy", mean_k) 

        step_norm = np.linalg.norm(mean_k, axis=1)  # (steps,)
        stats[f"layer_{k}"] = {
            "count": counts[k],
            "mean_step_norm_mean": float(step_norm.mean()),
            "mean_step_norm_std": float(step_norm.std())
        }
        if verbose:
            print(f"[OK] layer {k:02d} | used={counts[k]:3d} | step_norm μ={step_norm.mean():.3f} σ={step_norm.std():.3f}")

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return out_dir, stats

if __name__ == "__main__":
    path_act = "<Path for remain activation>"
    out, stats = make_layer_means(root_dir=path_act, out_subdir="_layer_means", verbose=True)
    print("Saved means to:", out)
    



