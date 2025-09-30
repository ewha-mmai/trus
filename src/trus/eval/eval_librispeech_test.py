# for SIM-R, WER-R, spk-ZRF-R, SIM-UF, WER-UF, spk-ZRF-UF

import argparse
import json
import os
import sys
import re
import glob
from pathlib import Path
import importlib.util
from importlib.resources import files

sys.path.insert(0, "F5-TTS/src")
sys.path.append(os.getcwd())

import multiprocessing as mp
import numpy as np
import torch

from f5_tts.eval.utils_eval import get_librispeech_test, run_asr_wer, run_sim
from f5_tts.eval.ecapa_tdnn import ECAPA_TDNN_SMALL

rel_path = str(files("f5_tts").joinpath("../../../"))

MOD_PATH = os.path.join(rel_path, "src", "f5_tts", "eval", "eval_spk_ZRF.py")

spec = importlib.util.spec_from_file_location("eval_spk_ZRF_new", MOD_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

build_testset_spkzrf = mod.build_testset_spkzrf
run_spkzrf = mod.run_spkzrf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--eval_task",
        type=str,
        default="wer",
        choices=["sim", "wer", "spk-ZRF"],
    )
    parser.add_argument("-l", "--lang", type=str, default="en")
    parser.add_argument("-g", "--gen_wav_dir", type=str, required=True,
                        help="Generated wav dir (θ for spk-ZRF)")
    parser.add_argument("-p", "--librispeech_test_clean_path", type=str, required=True,
                        help="LibriSpeech test-clean root, e.g., .../data/LibriSpeech/test-clean")
    parser.add_argument("--metalst", type=str, default=None,
                        help="Path to cross-sentence .lst (default: repo builtin)")
    parser.add_argument("-n", "--gpu_nums", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--local", action="store_true", help="Use local custom checkpoint directory")

    parser.add_argument("--gen_wav_dir_minus", type=str, default=None,
                        help="Generated wav dir for θ⁻ (text + speaker prompt)")
    parser.add_argument("--gen_name_tpl", type=str, default="{id}.wav")
    parser.add_argument("--skip_missing_gen", action="store_true")
    parser.add_argument("--sv_model_name", type=str, default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--sv_target_sr", type=int, default=16000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--alpha", type=float, default=5.5, help="temperature for sigmoid over cosine")
    parser.add_argument("--pooling", type=str, default="speaker",
                        choices=["utterance", "speaker"],

    return parser.parse_args()


UTT_RE = re.compile(r"\b(\d{1,6}-\d{1,6}-\d{1,6})\b")

def parse_ref_map_from_lst(metalst):
    m = {}
    with open(metalst, "r", encoding="utf-8") as f:
        for line in f:
            ref_utt, _, _, gen_utt, _, _ = line.strip().split("\t")
            m[gen_utt] = ref_utt
    return m

def utt_to_ref_flac(root: str, utt_id: str) -> str:
    spk, chap, _ = utt_id.split("-")
    return os.path.join(root, spk, chap, f"{utt_id}.flac")

def collect_utt_ids_from_theta(theta_dir: str) -> set[str]:
    ids = set()
    for p in glob.glob(os.path.join(theta_dir, "**", "*.wav"), recursive=True):
        stem = Path(p).stem
        if UTT_RE.fullmatch(stem):
            ids.add(stem)
    return ids

def collect_utt_ids_from_lst(metalst_path: str) -> set[str]:
    ids = set()
    with open(metalst_path, "r", encoding="utf-8") as f:
        for line in f:
            m = UTT_RE.search(line)
            if m:
                ids.add(m.group(1))
    return ids


def main():
    args = get_args()
    eval_task = args.eval_task
    lang = args.lang
    librispeech_test_clean_path = args.librispeech_test_clean_path
    gen_wav_dir = args.gen_wav_dir
    metalst = args.metalst or (rel_path + "/data/librispeech_pc_test_clean_cross_sentence.lst")
    print(f"[INFO] Using metalst: {metalst}")

    if eval_task == "spk-ZRF":
        if not args.gen_wav_dir_minus:
            raise ValueError("[spk-ZRF] --gen_wav_dir_minus is required (θ⁻ directory).")

        for d, name in [(gen_wav_dir, "gen_wav_dir (theta)"),
                        (args.gen_wav_dir_minus, "gen_wav_dir_minus (tminus)")]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"[spk-ZRF] {name} not found or not a directory: {d}")

        utt_ids = collect_utt_ids_from_theta(gen_wav_dir)
        if not utt_ids:
            utt_ids = collect_utt_ids_from_lst(metalst)
        if not utt_ids:
            raise RuntimeError("[spk-ZRF] No utt ids found from theta dir or .lst")

        ref_map = parse_ref_map_from_lst(metalst)

        items = []
        miss = 0
        for uid in sorted(utt_ids):
            ref_utt = ref_map.get(uid)
            if not ref_utt:
                miss += 1
                continue
            ref = utt_to_ref_flac(librispeech_test_clean_path, ref_utt)
            if not os.path.exists(ref):
                miss += 1
                continue
            spk_id = ref_utt.split("-")[0]
            items.append({
                "id": uid,
                "text": "",
                "ref_wav": ref,
                "spk_id": spk_id,
                "speaker": "", "language": "", "duration": 0.0, "dnsmos": None
            })

        if miss:
            print(f"[INFO] Missing ref flac for {miss} ids (skipped)")

        if not items:
            raise RuntimeError("[spk-ZRF] No valid items with existing enrollment flac found.")

        pairs = build_testset_spkzrf(
            items,
            gen_root_theta=gen_wav_dir,
            gen_root_tminus=args.gen_wav_dir_minus,
            skip_missing_gen=args.skip_missing_gen,
            gen_name_tpl=args.gen_name_tpl,
        )
        print(f"[INFO] Paired {len(pairs)} ids for spk-ZRF")

        device = torch.device("cpu") if args.cpu else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        results, spkzrf_mean = run_spkzrf(
            pairs,
            sv_model_name=args.sv_model_name,
            target_sr=args.sv_target_sr,
            batch_size=args.batch_size,
            device=device,
            use_amp=True, embed_batch_size=2, max_seconds=8.0,
            alpha=args.alpha,
            pooling = args.pooling
        )

        parent_dir = os.path.dirname(args.gen_wav_dir)
        os.makedirs(parent_dir, exist_ok=True)
        out_path = os.path.join(parent_dir, "_spkzrf_results.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.write(f"\nSPKZRF: {round(spkzrf_mean, 5)}\n")

        print(f"\nTotal {len(results)} samples")
        print(f"SPKZRF: {spkzrf_mean:.5f}")
        print(f"SPKZRF results saved to {out_path}")
        return

    gpus = list(range(args.gpu_nums))
    test_set = get_librispeech_test(metalst, gen_wav_dir, gpus, librispeech_test_clean_path)

    local = args.local
    if local:
        asr_ckpt_dir = os.path.join(rel_path, "checkpoints", "Systran", "faster-whisper-large-v3")
    else:
        asr_ckpt_dir = os.path.join(rel_path, "checkpoints", "Systran", "faster-whisper-large-v3")

    wavlm_ckpt_dir = os.path.join(rel_path, "checkpoints", "UniSpeech", "wavlm_large_finetune.pth")

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
        model.load_state_dict(state_dict["model"], strict=False)
        model = model.to(device)
        model.eval()

        args_list = [(rank, sub_test_set) for (rank, sub_test_set) in test_set]
        for a in args_list:
            r = run_sim(a, model, device)
            full_results.extend(r)

    else:
        raise ValueError(f"Unknown metric type: {eval_task}")

    parent_dir = os.path.dirname(gen_wav_dir) 
    os.makedirs(parent_dir, exist_ok=True)
    result_path = os.path.join(parent_dir, f"_{eval_task}_results.jsonl")
    with open(result_path, "w", encoding="utf-8") as f:
        for line in full_results:
            metrics.append(line[eval_task])
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        metric = round(np.mean(metrics), 5)
        f.write(f"\n{eval_task.upper()}: {metric}\n")

    print(f"\nTotal {len(metrics)} samples")
    print(f"{eval_task.upper()}: {metric}")
    print(f"{eval_task.upper()} results saved to {result_path}")


if __name__ == "__main__":
    main()
