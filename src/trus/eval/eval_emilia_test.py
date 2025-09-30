# for SIM-F, WER-F

import argparse
import json
import os
import sys
import glob
import re
import unicodedata
from pathlib import Path
from importlib.resources import files
import numpy as np
import torch

sys.path.insert(0, "F5-TTS/src")
sys.path.append(os.getcwd())

from f5_tts.eval.utils_eval import run_asr_wer, run_sim
from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL


PKG_DIR = Path(files("f5_tts").joinpath("")).resolve()
REL_ROOT = PKG_DIR.parent.parent
SRC_DIR = (REL_ROOT / "src").resolve()

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

sys.path.append(os.getcwd())


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--eval_task", type=str, default="wer", choices=["sim", "wer"])
    p.add_argument("-l", "--lang", type=str, default="en")
    p.add_argument("-g", "--gen_wav_dir", type=str, required=True, help="directory containing generated audios")
    p.add_argument("--jsonl_path", type=str, required=True, help="path to emilia_forget_test.jsonl")
    p.add_argument("--audio_root", type=str, required=True, help="root to prepend to 'wav' field to locate reference audio")
    p.add_argument("-n", "--gpu_nums", type=int, default=8, help="number of GPUs to partition the eval set for")
    p.add_argument("--local", action="store_true", help="use local custom checkpoint directory")
    p.add_argument("--min_dur", type=float, default=0.0, help="drop samples shorter than this (sec)")
    p.add_argument("--max_dur", type=float, default=0.0, help="drop samples longer than this (sec), 0=disabled")
    p.add_argument("--skip_missing_gen", action="store_true",
                   help="skip items whose generated audio isn't found instead of raising")
    p.add_argument("--gen_name_tpl", type=str, default="{id}.wav",
                   help="Generated filename template. Tokens: {id}, {sid}, {wid}, {num}. "
                        "Examples: '{sid}_{wid}.wav', '{wid}.wav', '{id}_pred.wav'")
    p.add_argument("--restrict_to_generated", action="store_true",
                   help="Evaluate only items that actually have a generated audio under --gen_wav_dir")
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
                alt = os.path.join(_clean(audio_root), f"{ex_id}.mp3")
                if os.path.exists(alt):
                    ref_abs = alt
                else:
                    print(f"[WARN] ref audio not found: {ref_abs} or {alt} -> skip '{ex_id}'")
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


def _split_round_robin(arr, gpus):
    buckets = [[] for _ in gpus]
    for i, x in enumerate(arr):
        buckets[i % len(gpus)].append(x)
    return [(rank, subset) for rank, subset in zip(gpus, buckets)]


def build_testset_wer(items, gen_root, gpus, skip_missing_gen=False, gen_name_tpl="{id}.wav"):
    triples = []
    for ex in items:
        utt_id = _clean(ex["id"])
        gen_path = find_gen_audio(gen_root, utt_id, gen_name_tpl)
        if gen_path is None:
            msg = f"[WARN] Generated audio not found for id={utt_id} under {gen_root}"
            if skip_missing_gen:
                print(msg + " -> skip")
                continue
            else:
                raise FileNotFoundError(msg)

        prompt_wav = ex["ref_wav"]
        truth_text = ex["text"]
        triples.append((gen_path, prompt_wav, truth_text))

    return _split_round_robin(triples, gpus)


def build_testset_sim(items, gen_root, gpus, skip_missing_gen=False, gen_name_tpl="{id}.wav"):
    triples = []
    for ex in items:
        utt_id = _clean(ex["id"])
        gen_path = find_gen_audio(gen_root, utt_id, gen_name_tpl)
        if gen_path is None:
            msg = f"[WARN] Generated audio not found for id={utt_id} under {gen_root}"
            if skip_missing_gen:
                print(msg + " -> skip")
                continue
            else:
                raise FileNotFoundError(msg)

        prompt_wav = ex["ref_wav"]
        truth_text = ex["text"]

        triples.append((gen_path, prompt_wav, truth_text))

    return _split_round_robin(triples, gpus)



def main():
    args = get_args()
    eval_task = args.eval_task
    lang = args.lang
    jsonl_path = args.jsonl_path
    audio_root = args.audio_root
    gen_wav_dir = args.gen_wav_dir

    raw_items = read_emilia_jsonl(
        jsonl_path=jsonl_path,
        audio_root=audio_root,
        lang_filter=lang,
        min_dur=args.min_dur,
        max_dur=args.max_dur,
    )

    if args.restrict_to_generated:
        kept = []
        for ex in raw_items:
            if find_gen_audio(gen_wav_dir, ex["id"], args.gen_name_tpl) is not None:
                kept.append(ex)
        print(f"[INFO] restrict_to_generated: {len(kept)}/{len(raw_items)} items kept")
        raw_items = kept

    gpus = list(range(args.gpu_nums))

    if eval_task == "wer":
        test_set = build_testset_wer(
            raw_items, gen_root=gen_wav_dir, gpus=gpus,
            skip_missing_gen=args.skip_missing_gen, gen_name_tpl=args.gen_name_tpl
        )
    else:  # sim
        test_set = build_testset_sim(
            raw_items, gen_root=gen_wav_dir, gpus=gpus,
            skip_missing_gen=args.skip_missing_gen, gen_name_tpl=args.gen_name_tpl
        )

    if args.local:
        asr_local_dir = (REL_ROOT / "checkpoints" / "Systran" / "faster-whisper-large-v3").resolve()
        if asr_local_dir.is_dir():
            asr_ckpt_dir = str(asr_local_dir)
        else:
            print(f"[WARN] ASR local dir not found: {asr_local_dir} -> fallback to HF Hub")
            asr_ckpt_dir = "Systran/faster-whisper-large-v3"
    else:
        asr_ckpt_dir = "Systran/faster-whisper-large-v3"
    
    wavlm_ckpt_dir = str((REL_ROOT / "checkpoints" / "UniSpeech" / "wavlm_large_finetune.pth").resolve())

    # Run evaluation
    full_results = []
    metrics = []

    if eval_task == "wer":
        args_list = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
        for a in args_list:
            r = run_asr_wer(a)
            full_results.extend(r)

    elif eval_task == "sim":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
        state_dict = torch.load(wavlm_ckpt_dir, map_location="cpu")
        model.load_state_dict(state_dict.get("model", state_dict), strict=False)
        model = model.to(device)
        model.eval()

        args_list = [(rank, sub_test_set) for (rank, sub_test_set) in test_set]
        for a in args_list:
            r = run_sim(a, model, device)
            full_results.extend(r)

    else:
        raise ValueError(f"Unknown metric type: {eval_task}")

    os.makedirs(gen_wav_dir, exist_ok=True)
    result_path = str(Path(gen_wav_dir) / f"_{eval_task}_results.jsonl")

    with open(result_path, "w", encoding="utf-8") as f:
        for line in full_results:
            metrics.append(line[eval_task])
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        metric = round(float(np.mean(metrics)), 5)
        f.write(f"\n{eval_task.upper()}: {metric}\n")

    print(f"\nTotal {len(metrics)} samples")
    print(f"{eval_task.upper()}: {metric}")
    print(f"{eval_task.upper()} results saved to {result_path}")


if __name__ == "__main__":
    main()
