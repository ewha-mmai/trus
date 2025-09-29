import os
import sys
import glob
from pathlib import Path
import json
import argparse
import time
from importlib.resources import files

sys.path.insert(0, "F5-TTS/src")
sys.path.append(os.getcwd())

import torch
import torchaudio
from accelerate import Accelerator
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm

from f5_tts.eval.utils_eval import (
    get_inference_prompt,
    get_librispeech_test_clean_metainfo,
    get_seedtts_testset_metainfo,
)
from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer

# ---------------- Consts ----------------
use_ema = True
target_rms = 0.1
MIN_FRAMES = 8
PROMPT_MIN = 3.0
PROMPT_MAX = 10.0
GT_MIN     = 1.2 
GT_MAX     = 6.0
TOTAL_MAX  = 40.0

accelerator = Accelerator()
device = accelerator.device

rel_path = "F5-TTS"

# ---------------- Utils ----------------
def _normalize_text_entry_for_text_only(t):
    if isinstance(t, str):
        s = t.strip()
        return s if s else "."
    if isinstance(t, (list, tuple)):
        parts = []
        stack = [t]
        while stack:
            x = stack.pop()
            if isinstance(x, (list, tuple)):
                stack.extend(reversed(x))
            elif isinstance(x, str):
                parts.append(x)
        s = " ".join(p for p in parts if p).strip()
        return s if s else "."
    return "."

def _safe_duration_sec(path: str):
    try:
        info = torchaudio.info(path)
        if info.num_frames and info.sample_rate:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception as e:
        print(f"[WARN] duration read failed: {path} ({e})")
    return None

def _spk_from_utt(utt_id: str):
    return utt_id.rsplit("_", 1)[0] if isinstance(utt_id, str) and "_" in utt_id else utt_id

def _resolve_path(wav_path: str, audio_root: str | None):
    if not wav_path:
        return wav_path
    p = Path(wav_path)
    if p.is_file():
        return str(p)
    if audio_root:
        cand = Path(audio_root) / wav_path
        return str(cand)
    return wav_path

def _read_emilia_jsonl_paired(jsonl_path: str, audio_root: str | None = None):
    items = []
    by_spk = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                utt_id = obj.get("id") or obj.get("utt")
                wav_path = _resolve_path(obj.get("wav", ""), audio_root)
                text = (obj.get("text") or "").strip()
                spk = obj.get("speaker") or ( _spk_from_utt(utt_id) if utt_id else None )
                dur = obj.get("duration", None)
                if dur is None and wav_path:
                    dur = _safe_duration_sec(wav_path)
                if not utt_id or not wav_path or spk is None:
                    print(f"[WARN] missing keys -> skip: {obj}")
                    continue
                item = {"utt": utt_id, "spk": spk, "wav": wav_path, "text": text, "dur": dur}
                items.append(item)
                by_spk.setdefault(spk, []).append(item)
            except Exception as e:
                print(f"[WARN] jsonl parse failed -> skip: {line} ({e})")

    metainfo = []
    miss_ref = 0
    for tgt in items:
        tgt_d = tgt["dur"]
        if tgt_d is None:
            print(f"[INFO] GT duration unknown -> skip: {tgt['utt']}")
            continue

        cands = [
            x for x in by_spk.get(tgt["spk"], [])
            if x["utt"] != tgt["utt"] and x["dur"] is not None and (PROMPT_MIN <= x["dur"] <= PROMPT_MAX)
        ]

        prompt = None
        if cands:
            prompt = min(cands, key=lambda x: abs(x["dur"] - PROMPT_MIN))
        else:
            if (tgt_d >= PROMPT_MIN) and (2 * tgt_d <= TOTAL_MAX):
                prompt = tgt
            else:
                miss_ref += 1
                continue

        total = (prompt["dur"] or 0.0) + (tgt_d or 0.0)
        if total > TOTAL_MAX:
            continue

        utt          = tgt["utt"]
        prompt_text  = prompt["text"]
        prompt_wav   = prompt["wav"]

        gt_text      = tgt["text"] if tgt["text"] else "."
        gt_wav       = tgt["wav"]

        metainfo.append((utt, prompt_text, prompt_wav, gt_text, gt_wav))

    print(f"[INFO] Emilia pairing done: {len(metainfo)} samples (ref-miss {miss_ref})")
    return metainfo


def main():
    parser = argparse.ArgumentParser(description="batch inference")

    parser.add_argument("-s", "--seed", default=None, type=int)
    parser.add_argument("-n", "--expname", required=True)
    parser.add_argument("-c", "--ckptstep", default=1250000, type=int)

    parser.add_argument("-nfe", "--nfestep", default=32, type=int)
    parser.add_argument("-o", "--odemethod", default="euler")
    parser.add_argument("-ss", "--swaysampling", default=-1, type=float)

    parser.add_argument("-t", "--testset", required=True, choices=["ls_pc_test_clean", "emilia"],
                        help="Choose between LibriSpeech pc-cross config and Emilia(jsonl) config.")
    parser.add_argument("--metalst", type=str, default=None,
                        help="Path to cross-sentence .lst (overrides default).")
    parser.add_argument("--ls_root", type=str, default=None,
                        help="LibriSpeech root for wav/flac lookup (overrides default test-clean).")

    parser.add_argument("--no_ref_audio", action="store_true",
                        help="Generate with TEXT ONLY by dropping audio conditioning at inference.")
    parser.add_argument("--seed_per_utt", action="store_true",
                        help="Use a different random seed per utterance to encourage speaker randomization.")
    parser.add_argument("--emilia_jsonl", type=str, default=None,
                        help="Path to Emilia-style jsonl (id, wav, text, duration, speaker, ...).")
    parser.add_argument("--audio_root", type=str, default=None,
                        help="Root directory to resolve relative paths inside jsonl.")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Override checkpoint path (use .pt or .safetensors directly).")

    args = parser.parse_args()

    seed = args.seed
    exp_name = args.expname
    ckpt_step = args.ckptstep
    nfe_step = args.nfestep
    ode_method = args.odemethod
    sway_sampling_coef = args.swaysampling
    testset = args.testset

    infer_batch_size = 1
    cfg_strength = 2.0
    speed = 1.0
    use_truth_duration = False
    no_ref_audio = args.no_ref_audio
    seed_per_utt = args.seed_per_utt

    print("[DBG] before OmegaConf.load")
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{exp_name}.yaml")))
    print("[DBG] after OmegaConf.load")
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch

    dataset_name = model_cfg.datasets.name
    tokenizer = model_cfg.model.tokenizer

    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
    hop_length = model_cfg.model.mel_spec.hop_length
    win_length = model_cfg.model.mel_spec.win_length
    n_fft = model_cfg.model.mel_spec.n_fft

    if testset == "ls_pc_test_clean":
        metalst = args.metalst or (rel_path + "/data/librispeech_pc_test_clean_cross_sentence.lst")
        librispeech_test_clean_path = args.ls_root or "data/LibriSpeech/test-clean"
        print(f"[INFO] LibriSpeech meta: metalst={metalst}")
        print(f"[INFO] LibriSpeech root: {librispeech_test_clean_path}")
        metainfo = get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path)

    elif testset == "emilia":
        assert args.emilia_jsonl is not None, "--emilia_jsonl is required when -t emilia"
        metainfo = _read_emilia_jsonl_paired(args.emilia_jsonl, args.audio_root)

    gt_text_map = None
    if 'metainfo' in locals() and metainfo is not None:
        gt_text_map = {utt: gt_text for (utt, _pt, _pw, gt_text, _gw) in metainfo}

    # -------- output dir --------
    subdir_prefix = ""
    if args.metalst is not None:
        subdir_prefix = "unseen/"

    output_dir = (
        f"{rel_path}/"
        f"results/{exp_name}_{ckpt_step}/{subdir_prefix}{testset}/"
        f"seed{seed}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
        f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
        f"_cfg{cfg_strength}_speed{speed}"
        f"{'_gt-dur' if use_truth_duration else ''}"
        f"{'_no-ref-audio' if no_ref_audio else ''}"
    )

    # -------- prompts --------
    print("[DBG] before get_inference_prompt")
    prompts_all = get_inference_prompt(
        metainfo,
        speed=speed,
        tokenizer=tokenizer,
        target_sample_rate=target_sample_rate,
        n_mel_channels=n_mel_channels,
        hop_length=hop_length,
        mel_spec_type=mel_spec_type,
        target_rms=target_rms,
        use_truth_duration=use_truth_duration,
        infer_batch_size=infer_batch_size,
    )
    print("[DBG] after get_inference_prompt")

    # -------- vocoder --------
    local = False
    if mel_spec_type == "vocos":
        vocoder_local_path = "../checkpoints/charactr/vocos-mel-24khz"
    elif mel_spec_type == "bigvgan":
        vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    print(f"[DBG] mel_spec_type={model_cfg.model.mel_spec.mel_spec_type}")
    print("[DBG] before load_vocoder")
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=local, local_path=vocoder_local_path)
    print("[DBG] after load_vocoder")

    # -------- tokenizer & model --------
    vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    # -------- checkpoint --------
    if args.ckpt_path is not None:
        ckpt_path = args.ckpt_path
        print(f'[info] using ckpt override: {ckpt_path}')
    else:
        ckpt_prefix = rel_path + f"/ckpts/{exp_name}/model_{ckpt_step}"
        print(f'{ckpt_prefix=}')
        if os.path.exists(ckpt_prefix + ".pt"):
            ckpt_path = ckpt_prefix + ".pt"
        elif os.path.exists(ckpt_prefix + ".safetensors"):
            ckpt_path = ckpt_prefix + ".safetensors"
        else:
            print("Loading from self-organized training checkpoints rather than released pretrained.")
            ckpt_path = rel_path + f"/{model_cfg.ckpts.save_dir}/model_{ckpt_step}.pt"
    print(f'{ckpt_path=}')
    dtype = torch.float32 if mel_spec_type == "bigvgan" else None
    model = load_checkpoint(model, ckpt_path, "cpu", dtype=dtype, use_ema=use_ema).to(device)

    if not os.path.exists(output_dir) and accelerator.is_main_process:
        os.makedirs(output_dir)

    # -------- inference --------
    accelerator.wait_for_everyone()
    start = time.time()

    with accelerator.split_between_processes(prompts_all) as prompts:
        for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
            utts, ref_rms_list, ref_mels, ref_mel_lens, total_mel_lens, final_text_list = prompt

            if no_ref_audio:
                if gt_text_map is not None:
                    try:
                        final_text_list = [gt_text_map.get(u, final_text_list[j]) for j, u in enumerate(utts)]
                    except Exception as e:
                        if accelerator.is_local_main_process:
                            print(f"[WARN] text-only gt override failed: {e}")
                final_text_list = [_normalize_text_entry_for_text_only(t) for t in final_text_list]

            ref_mels = ref_mels.to(device)
            ref_mel_lens = torch.tensor(ref_mel_lens, dtype=torch.long, device=device)
            total_mel_lens = torch.tensor(total_mel_lens, dtype=torch.long, device=device)

            with torch.inference_mode():
                if seed_per_utt:
                    for i in range(len(utts)):
                        seed_i = (seed if seed is not None else 0) + (hash(utts[i]) % 1_000_000_007)
                        lens_i = torch.zeros_like(ref_mel_lens[i:i+1]) if no_ref_audio else ref_mel_lens[i:i+1]
                        if no_ref_audio:
                            dur_i = (total_mel_lens[i:i+1] - ref_mel_lens[i:i+1]).clamp_min(MIN_FRAMES)
                        else:
                            dur_i = total_mel_lens[i:i+1]

                        gen_i, _ = model.sample(
                            cond=ref_mels[i:i+1, ...],
                            text=[final_text_list[i]] if isinstance(final_text_list[i], str) else final_text_list[i],
                            duration=dur_i,
                            lens=lens_i,
                            steps=nfe_step,
                            cfg_strength=cfg_strength,
                            sway_sampling_coef=sway_sampling_coef,
                            no_ref_audio=no_ref_audio,
                            seed=seed_i,
                        )
                        gen = gen_i[0]

                        if no_ref_audio:
                            end_i = int((total_mel_lens[i] - ref_mel_lens[i]).item())
                            gen = gen[: end_i, :].unsqueeze(0)
                        else:
                            start_idx = int(ref_mel_lens[i].item())
                            end_i = int(total_mel_lens[i].item())
                            if end_i <= start_idx:
                                end_i = start_idx + MIN_FRAMES
                            gen = gen[start_idx:end_i, :].unsqueeze(0)

                        if gen.shape[1] < MIN_FRAMES:
                            if accelerator.is_local_main_process:
                                print(f"[WARN] too few mel frames ({gen.shape[1]}), skip utt={utts[i]}")
                            continue

                        gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
                        if mel_spec_type == "vocos":
                            wave = vocoder.decode(gen_mel_spec).cpu()
                        else:
                            wave = vocoder(gen_mel_spec).squeeze(0).cpu()
                        rr = float(ref_rms_list[i])
                        if rr < target_rms:
                            wave = wave * (rr / target_rms)
                        torchaudio.save(f"{output_dir}/{utts[i]}.wav", wave, target_sample_rate)

                else:
                    lens_to_use = torch.zeros_like(ref_mel_lens) if no_ref_audio else ref_mel_lens
                    duration_to_use = (
                        (total_mel_lens - ref_mel_lens).clamp_min(MIN_FRAMES)
                        if no_ref_audio else total_mel_lens
                    )

                    generated, _ = model.sample(
                        cond=ref_mels,
                        text=[(t if isinstance(t, str) else t) for t in final_text_list],
                        duration=duration_to_use,
                        lens=lens_to_use,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                        no_ref_audio=no_ref_audio,
                        seed=seed,
                    )

                    for i, gen in enumerate(generated):
                        if no_ref_audio:
                            end_i = int((total_mel_lens[i] - ref_mel_lens[i]).item())
                            gen = gen[: end_i, :].unsqueeze(0)
                        else:
                            start_idx = int(lens_to_use[i].item())
                            end_i = int(duration_to_use[i].item())
                            if end_i <= start_idx:
                                end_i = start_idx + MIN_FRAMES
                            gen = gen[start_idx:end_i, :].unsqueeze(0)

                        if gen.shape[1] < MIN_FRAMES:
                            if accelerator.is_local_main_process:
                                print(f"[WARN] too few mel frames ({gen.shape[1]}), skip utt={utts[i]}")
                            continue

                        gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
                        if mel_spec_type == "vocos":
                            generated_wave = vocoder.decode(gen_mel_spec).cpu()
                        else:
                            generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()

                        rr = float(ref_rms_list[i])
                        if rr < target_rms:
                            generated_wave = generated_wave * (rr / target_rms)

                        if isinstance(generated_wave, torch.Tensor) and generated_wave.ndim == 1:
                            generated_wave = generated_wave.unsqueeze(0)
                        if generated_wave.shape[-1] == 0:
                            if accelerator.is_local_main_process:
                                print(f"[WARN] empty waveform; skip save utt={utts[i]}")
                            continue
                        torchaudio.save(f"{output_dir}/{utts[i]}.wav", generated_wave, target_sample_rate)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        timediff = time.time() - start
        print(f"Done batch inference in {timediff / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
