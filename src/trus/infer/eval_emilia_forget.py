#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json
from pathlib import Path
from typing import Optional, Dict, List, Tuple


import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm
import torchaudio

# ======================= path & global setting =======================
DEVICE                 = "cuda:0"
NFE_STEP               = 32
ALPHA                  = 1.5
DEBUG_HOOK             = True
RESAMPLE_DIFFS_TO_NFE  = True
ALLOW_DIM_PROJ         = True
FORCE_NOISE_TEST       = False   # If True, inject small amount of noise instead of erase_proj. Smoke test.

# --------- data/model path ---------
REMAIN_NUM       = 50  # the number of remain speakers : {10, 30, 50} 
FORGET_ROOT      = Path("<Path for forget activation>")
REMAIN_MEAN_DIR  = Path(f"<Path for remain_{REMAIN_NUM}_mean activation>")

DIFF_ROOT        = Path(f"<Path for diff_emilia_remain_{REMAIN_NUM}")

# model/vocoder
VOCODER_NAME      = "vocos"             # "vocos" | "bigvgan"
LOAD_VOC_LOCAL    = False
MODEL_CFG_YAML    = Path("<Path for configs/F5TTS_v1_Base.yaml>")
CKPT_FILE         = Path("<Path for ckpts/F5TTS_v1_Base/model_1250000.safetensors>")
VOCAB_TXT         = ""

# filename pattern
LAYER_PAT_FORGET = re.compile(r"_layer_(\d+)\.npy$", re.IGNORECASE)
LAYER_PAT_REMAIN = re.compile(r"remain_\d+_mean_layer_(\d+)\.npy$", re.IGNORECASE)

# ======================= for Ablation =======================
# layer band mode : "lt_mu_minus_sigma" | "lt_mu" | "lt_mu_plus_sigma" | "all"
LAYER_BAND_MODE   = "lt_mu_plus_sigma"
LAYER_K           = 1.0         # σ multiplier (bandwidth factor)
DROP_WORST        = False       

# step selection rule: "lt_layer_mean" | "lt_layer_mu"
STEP_RULE         = "lt_layer_mean"

# output path
RUN_TAG_BASE      = ""
RUN_TAG           = f"{RUN_TAG_BASE}_r{REMAIN_NUM}_{ALPHA}"
STEERED_OUT_DIR   = Path(f"<Path for /results/{RUN_TAG}_{LAYER_BAND_MODE}>")

# Save selection results
SAVE_PICKED_JSON  = True
RUN_TAG           = f"{RUN_TAG}_{LAYER_BAND_MODE}_{STEP_RULE}"

# ======================= F5-TTS util =======================
from trus.infer.utils_infer import (
    infer_process, load_model, load_vocoder,
    target_rms as _TARGET_RMS, cross_fade_duration as _XF,
    nfe_step as _NFE_DEFAULT, cfg_strength as _CFG,
    sway_sampling_coef as _SWAY, speed as _SPEED, fix_duration as _FIXDUR
)

# ======================= helper utils =======================
def _resolve_path(wav_path: str, audio_root: str | None):
    p = Path(wav_path)
    if p.is_file():
        return str(p)
    return str(Path(audio_root) / wav_path) if audio_root else wav_path

def _safe_duration_sec(path: str):
    try:
        info = torchaudio.info(path)
        if info.num_frames and info.sample_rate:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass
    return None

def _spk_from_utt(utt_id: str):
    return utt_id.rsplit("_", 1)[0] if isinstance(utt_id, str) and "_" in utt_id else utt_id

def read_emilia_jsonl_paired(jsonl_path: str, audio_root: str | None = None) -> List[Tuple[str,str,str,str,str]]:
    
    PROMPT_MIN, PROMPT_MAX, TOTAL_MAX = 3.0, 10.0, 40.0
    items, by_spk = [], {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line: continue
            obj = json.loads(line)
            utt_id  = obj.get("id") or obj.get("utt")
            wav     = _resolve_path(obj.get("wav",""), audio_root)
            text    = (obj.get("text") or "").strip()
            spk     = obj.get("speaker") or (_spk_from_utt(utt_id) if utt_id else None)
            dur     = obj.get("duration", None)
            if dur is None and wav: dur = _safe_duration_sec(wav)
            if not utt_id or not wav or spk is None: continue
            item={"utt":utt_id,"spk":spk,"wav":wav,"text":text,"dur":dur}
            items.append(item)
            by_spk.setdefault(spk,[]).append(item)
    meta, miss_ref = [], 0
    for tgt in items:
        tgt_d = tgt["dur"]
        if tgt_d is None: continue
        cands = [x for x in by_spk.get(tgt["spk"],[])
                 if x["utt"]!=tgt["utt"] and x["dur"] and (PROMPT_MIN<=x["dur"]<=PROMPT_MAX)]
        if cands:
            prompt = min(cands, key=lambda x: abs((x["dur"] or PROMPT_MIN)-PROMPT_MIN))
        else:
            if (tgt_d>=PROMPT_MIN) and (2*tgt_d<=TOTAL_MAX):
                prompt = tgt
            else:
                miss_ref += 1
                continue
        total = (prompt["dur"] or 0.0) + (tgt_d or 0.0)
        if total>TOTAL_MAX: continue
        meta.append((tgt["utt"], prompt["text"], prompt["wav"], tgt["text"] or ".", tgt["wav"]))
    print(f"[INFO] Emilia pairing done: {len(meta)} samples (ref-miss {miss_ref})")
    return meta

def map_layers(dirpath: Path, pat: re.Pattern) -> Dict[int, Path]:
    mp = {}
    if not dirpath.is_dir():
        return mp
    for f in os.listdir(dirpath):
        m = pat.search(f)
        if m:
            mp[int(m.group(1))] = dirpath / f
    return dict(sorted(mp.items()))

def _resample_steps_np(arr: np.ndarray, target_T: int) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr[None, :]
    T, D = arr.shape
    if T == target_T:
        return arr.astype(np.float32, copy=False)
    xs = np.linspace(0.0, 1.0, T, dtype=np.float32)
    xt = np.linspace(0.0, 1.0, target_T, dtype=np.float32)
    out = np.empty((target_T, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(xt, xs, arr[:, d])
    return out

def diag_cos(F: np.ndarray, R: np.ndarray, eps: float=1e-8) -> np.ndarray:
    
    if F.ndim == 1: F = F[None, :]
    if R.ndim == 1: R = R[None, :]
    T = min(F.shape[0], R.shape[0])
    F = F[:T]; R = R[:T]
    num = (F * R).sum(axis=1)
    den = (np.linalg.norm(F,axis=1) * np.linalg.norm(R,axis=1) + eps)
    return (num / den).astype(np.float32)

# ======================= diff loader =======================
def load_diffs_for_sample(sample_diff_dir: Path, layers: List[int], nfe_steps: int) -> Dict[int, np.ndarray]:
    d = {}
    sid = sample_diff_dir.name
    for li in layers:
        cands = [
            sample_diff_dir / f"{sid}_layer_{li}_diff_unit2d.npy",
            sample_diff_dir / f"layer_{li}_diff_unit2d.npy",
            sample_diff_dir / f"{sid}_layer_{li}_diff.npy",
            sample_diff_dir / f"layer_{li}_diff.npy",
        ]
        arr = None
        for p in cands:
            if p.is_file():
                arr = np.load(str(p)); break
        if arr is None:
            st = sample_diff_dir / "diff_stack.npy"
            if st.is_file():
                st_arr = np.load(str(st))
                if st_arr.ndim == 3 and 1 <= li <= st_arr.shape[0]:
                    arr = st_arr[li - 1]
        if arr is not None:
            if arr.ndim == 1: arr = arr[None, :]
            arr = arr.astype(np.float32, copy=False)
            if RESAMPLE_DIFFS_TO_NFE:
                arr = _resample_steps_np(arr, nfe_steps)
            d[li] = arr
    return d

# ======================= DiT & step context =======================
def find_dit_and_blocks(model: nn.Module):
    for m in model.modules():
        if hasattr(m, "transformer_blocks") and isinstance(m.transformer_blocks, nn.ModuleList):
            return m, list(m.transformer_blocks)
    raise RuntimeError("DiT not found")

class StepCtx:
    def __init__(self, steps: int):
        self.steps = steps
        self.current_step = 0
        self._call_idx = 0

def make_time_tap(ctx: StepCtx, steps=32):
    def hook(module, args):
        s = None
        if args and len(args) >= 4:
            try:
                t = args[3]
                t0 = float(t.flatten()[0].item()) if hasattr(t, "flatten") else float(t)
                if 0.0 <= t0 <= 1.0: s = int(t0 * steps)
            except: pass
        if s is None:
            s = ctx._call_idx
        ctx.current_step = max(0, min(steps - 1, s))
        ctx._call_idx = max(0, min(steps - 1, ctx._call_idx + 1))
    return hook

# ======================= layer selection rule 2 : variance =======================
def pick_layers_by_band(layer_mean_cos: Dict[int, float], mode: str, k_sigma: float, drop_worst: bool):
    vals = np.array(list(layer_mean_cos.values()), dtype=np.float64)
    mu   = float(vals.mean())
    sig  = float(vals.std(ddof=0))
    if mode == "lt_mu_minus_sigma":
        thr = mu - k_sigma * sig
        cand = [li for li, m in layer_mean_cos.items() if m < thr]
    elif mode == "lt_mu":
        cand = [li for li, m in layer_mean_cos.items() if m < mu]
    elif mode == "lt_mu_plus_sigma":
        thr = mu + k_sigma * sig
        cand = [li for li, m in layer_mean_cos.items() if m < thr]
    elif mode == "all":
        cand = sorted(list(layer_mean_cos.keys()))
    else:
        raise ValueError(f"Unknown LAYER_BAND_MODE={mode}")

    if drop_worst and len(cand) > 1:
        worst = min(cand, key=lambda li: layer_mean_cos[li])
        cand = [li for li in cand if li != worst]
    return sorted(cand), mu, sig

def pick_steps_for_layer(cs: np.ndarray, rule: str) -> np.ndarray:
    # cs: (T,) step-wise cos of a layer
    if rule == "lt_layer_mean":
        thr = float(np.mean(cs))
        keep = np.where(cs < thr)[0]
    elif rule == "lt_layer_mu":
        mu = float(np.mean(cs))
        sig = float(np.std(cs, ddof=0))
        thr = mu - sig
        keep = np.where(cs < thr)[0]
    else:
        raise ValueError(f"Unknown STEP_RULE={rule}")
    return keep

# ======================= erase_proj logic =======================
class StepwiseLayerSteering:
    def __init__(self, targets: List[nn.Module], diffs: Dict[int, np.ndarray],
                 alpha: float, ctx: 'StepCtx', layer_ids: List[int],
                 device=None, inject_steps_map: Optional[Dict[int, set]] = None):
        self.targets = targets
        self.alpha = float(alpha)
        self.ctx = ctx
        self.layer_ids = layer_ids
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.inject_steps_map = inject_steps_map or {}
        self._hooks = []
        self._diffs: Dict[int, torch.Tensor] = {}
        self._units: Dict[int, torch.Tensor] = {}
        self.stats = {"calls":0, "applied":0, "skips":{}, "per_layer_calls":{}, "per_layer_applied":{}}

        eps = 1e-8
        for li, arr in diffs.items():
            t = torch.from_numpy(arr).to(self.device).float()   # (T,D)
            if t.ndim == 1: t = t.unsqueeze(0)
            self._diffs[li] = t
            self._units[li] = t / (t.norm(p=2, dim=-1, keepdim=True) + eps)

        self._proj_cache: Dict[Tuple[int,int,int], torch.Tensor] = {}

    def _count(self, key, inc=1):
        self.stats[key] = self.stats.get(key, 0) + inc

    def _mk(self, d, li):
        d[li] = d.get(li, 0) + 1

    def _sk(self, reason):
        self.stats["skips"][reason] = self.stats["skips"].get(reason, 0) + 1

    def _make_hook(self, li:int):
        def hook(mod, inp, out):
            self._count("calls"); self._mk(self.stats["per_layer_calls"], li)

            if li not in self._diffs:
                self._sk("no_diff"); return out

            if self.ctx is None or self.ctx.steps is None:
                s = 0
            else:
                s = int(max(0, min(self.ctx.steps - 1, getattr(self.ctx, "current_step", 0))))

            allowed = self.inject_steps_map.get(li, None)
            if allowed is not None and (s not in allowed):
                self._sk(f"step_gate_li{li}"); return out

            y = out
            if not torch.is_tensor(y):
                self._sk("out_not_tensor"); return out

            if torch.isnan(y).any() or torch.isinf(y).any():
                y = torch.nan_to_num(y, 0.0, 1e4, -1e4)

            C = y.shape[-1]
            T = self._diffs[li].shape[0]
            s = max(0, min(T - 1, s))

            v = self._diffs[li][s]  # (C,)
            if v.numel() != C:
                if not ALLOW_DIM_PROJ:
                    self._sk(f"C_mismatch_{v.numel()}to{C}"); return out
                key = (li, int(v.numel()), int(C))
                if key not in self._proj_cache:
                    W = torch.empty(v.numel(), C, device=self.device, dtype=y.dtype)
                    torch.nn.init.orthogonal_(W)
                    self._proj_cache[key] = W
                v = (v @ self._proj_cache[key]).contiguous()

            a = self.alpha
            if a == 0.0 and not FORCE_NOISE_TEST:
                self._sk("alpha_zero"); return out

            v = v.to(y.dtype).view(*([1]*(y.dim()-1)), C)

            if FORCE_NOISE_TEST:
                y_new = y + torch.randn_like(v) * 0.01
                if DEBUG_HOOK:
                    print(f"[SMOKE] li={li} step={s} add_noise=0.01 |y_rms|={float(y.pow(2).mean().sqrt()):.4g}")
                self._count("applied"); self._mk(self.stats["per_layer_applied"], li)
                return y_new

            # ---- erase_proj only ----
            u = self._units[li][s]
            u = (u / (u.norm(p=2) + 1e-8)).to(y.dtype).view(*([1]*(y.dim()-1)), C)
            proj = (y * u).sum(dim=-1, keepdim=True) * u
            y_new = y - a * proj

            if DEBUG_HOOK:
                # pr = float(proj.pow(2).mean().sqrt().item()); y_rms = float(y.pow(2).mean().sqrt().item())
                pass

            self._count("applied"); self._mk(self.stats["per_layer_applied"], li)
            return y_new
        return hook

    def attach(self):
        for li, mod in zip(self.layer_ids, self.targets):
            self._hooks.append(mod.register_forward_hook(self._make_hook(li)))

    def detach(self):
        for h in self._hooks:
            try: h.remove()
            except: pass
        self._hooks.clear()

# ======================= model loader =======================
def load_everything():
    if VOCODER_NAME == "vocos":
        voc_local = "../checkpoints/vocos-mel-24khz"
    else:
        voc_local = "../checkpoints/bigvgan_v2_24khz_100band_256x"
    voc = load_vocoder(VOCODER_NAME, is_local=LOAD_VOC_LOCAL, local_path=voc_local, device=DEVICE)

    model_cfg = OmegaConf.load(str(MODEL_CFG_YAML))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    ckpt = str(CKPT_FILE)
    ema = load_model(model_cls, model_arc, ckpt, mel_spec_type=VOCODER_NAME, vocab_file=VOCAB_TXT, device=DEVICE)
    return ema, voc

# ======================= main =======================
def main():
    STEERED_OUT_DIR.mkdir(parents=True, exist_ok=True)

    ema, voc = load_everything()
    dit, blocks = find_dit_and_blocks(ema)

    # Emilia meta load
    EMILIA_JSONL = "<Path for _script_for_data/Emilia_EN_jsonl_modi_path/Emilia_EN_forget_test_modi_path.jsonl>"
    EMILIA_ROOT  = "<Path for Emilia audio dir>"
    meta = read_emilia_jsonl_paired(EMILIA_JSONL, EMILIA_ROOT)
    print(f"[INFO] Emilia meta loaded: {len(meta)} samples")

    ok_cnt, skip_cnt = 0, 0

    for (sid, prompt_text, prompt_wav, gt_text, gt_wav) in tqdm(meta, desc="[STEER-EMILIA-ABL]"):
        # (A) forget / remain_mean 
        f_dir = FORGET_ROOT / sid
        if not f_dir.is_dir():
            print(f"[WARN] forget dir missing: {f_dir}; skip"); skip_cnt += 1; continue
        f_layers = map_layers(f_dir, LAYER_PAT_FORGET)
        r_layers = map_layers(REMAIN_MEAN_DIR, LAYER_PAT_REMAIN)
        common_layers = sorted(set(f_layers) & set(r_layers))
        if not common_layers:
            print(f"[WARN] no common layers for {sid}; skip"); skip_cnt += 1; continue

        layer_mean_cos: Dict[int, float] = {}
        layer_step_cos: Dict[int, np.ndarray] = {}
        for li in common_layers:
            F = np.load(str(f_layers[li])); R = np.load(str(r_layers[li]))
            if RESAMPLE_DIFFS_TO_NFE:
                F = _resample_steps_np(F, NFE_STEP)
                R = _resample_steps_np(R, NFE_STEP)
            cs = diag_cos(F, R)                 # (T,)
            layer_step_cos[li] = cs
            layer_mean_cos[li] = float(np.mean(cs))

        if not layer_mean_cos:
            print(f"[WARN] no step-wise cos for {sid}; skip"); skip_cnt += 1; continue

        selected_layers, mu_layers, sigma_layers = pick_layers_by_band(
            layer_mean_cos, mode=LAYER_BAND_MODE, k_sigma=LAYER_K, drop_worst=DROP_WORST
        )
        if not selected_layers:
            print(f"[INFO] no layer selected by {LAYER_BAND_MODE} for {sid}; skip")
            skip_cnt += 1; continue

        inject_steps_map: Dict[int, set] = {}
        for li in sorted(selected_layers):
            cs = layer_step_cos[li]             # (T,)
            keep_idx = pick_steps_for_layer(cs, rule=STEP_RULE)
            if keep_idx.size > 0:
                inject_steps_map[li] = set(int(x) for x in keep_idx.tolist())

        final_layers = [li for li in selected_layers if li in inject_steps_map and len(inject_steps_map[li]) > 0]
        if not final_layers:
            print(f"[INFO] no (layer,step) after step-rule {STEP_RULE} for {sid}; skip"); skip_cnt += 1; continue

        sample_diff_dir = DIFF_ROOT / sid
        if not sample_diff_dir.is_dir():
            alt = DIFF_ROOT / Path(gt_wav).stem
            if alt.is_dir():
                sample_diff_dir = alt
        if not sample_diff_dir.is_dir():
            print(f"[WARN] diff dir missing for {sid}: {sample_diff_dir}, skip"); skip_cnt += 1; continue

        diffs = load_diffs_for_sample(sample_diff_dir, final_layers, NFE_STEP)
        if not diffs:
            print(f"[WARN] diffs missing for selected layers {final_layers} at {sample_diff_dir}; skip")
            skip_cnt += 1; continue

        step_ctx = StepCtx(steps=NFE_STEP)
        time_hook = dit.register_forward_pre_hook(make_time_tap(step_ctx, steps=NFE_STEP))

        max_li = len(blocks)
        targets, valid_ids = [], []
        for li in final_layers:
            if 1 <= li <= max_li:
                targets.append(blocks[li - 1])  
                valid_ids.append(li)
        if not valid_ids:
            print(f"[WARN] valid layers empty(range) for {sid}; skip")
            try: time_hook.remove()
            except: pass
            skip_cnt += 1; continue

        steerer = StepwiseLayerSteering(
            targets=targets,
            diffs={li: diffs[li] for li in valid_ids if li in diffs},
            alpha=ALPHA,
            ctx=step_ctx,
            layer_ids=valid_ids,
            device=torch.device(DEVICE) if isinstance(DEVICE, str) else DEVICE,
            inject_steps_map=inject_steps_map,   # per-layer step set
        )
        steerer.attach()

        ref_audio = _resolve_path(prompt_wav, EMILIA_ROOT)
        ref_text_p = (prompt_text or ".")
        gen_text   = (gt_text or ".")
        try:
            wav, sr, _ = infer_process(
                ref_audio, ref_text_p, gen_text, ema, voc,
                mel_spec_type=VOCODER_NAME, target_rms=_TARGET_RMS,
                cross_fade_duration=_XF, nfe_step=NFE_STEP, cfg_strength=_CFG,
                sway_sampling_coef=_SWAY, speed=_SPEED, fix_duration=_FIXDUR, device=DEVICE
            )
        finally:
            if DEBUG_HOOK:
                print(f"[STAT] calls={steerer.stats['calls']} applied={steerer.stats['applied']} skips={steerer.stats['skips']}")
                print(f"[STAT] per_layer_calls={steerer.stats['per_layer_calls']}")
                print(f"[STAT] per_layer_applied={steerer.stats['per_layer_applied']}")
            steerer.detach()
            try: time_hook.remove()
            except: pass

        STEERED_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_wav = STEERED_OUT_DIR / f"{sid}.wav"
        sf.write(str(out_wav), wav, sr)
        ok_cnt += 1

        if SAVE_PICKED_JSON:
            picked = {int(li): sorted(list(inject_steps_map[li])) for li in valid_ids}
            dump = {
                "sid": sid,
                "remain_num": REMAIN_NUM,
                "layer_band_mode": LAYER_BAND_MODE,
                "layer_k": LAYER_K,
                "drop_worst": DROP_WORST,
                "step_rule": STEP_RULE,
                "alpha": ALPHA,
                "mu_layers": mu_layers,
                "sigma_layers": sigma_layers,
                "selected_layers": [int(li) for li in selected_layers],
                "final_layers": [int(li) for li in valid_ids],
                "layer_mean_cos": {int(li): float(layer_mean_cos[li]) for li in selected_layers},
                "picked_steps": picked
            }
            pj = STEERED_OUT_DIR / "picked_json"
            pj.mkdir(parents=True, exist_ok=True)
            with open(pj / f"{sid}.{RUN_TAG}.json", "w", encoding="utf-8") as f:
                json.dump(dump, f, ensure_ascii=False, indent=2)

        if DEBUG_HOOK:
            print(f"[OK] {out_wav.name} | layers={valid_ids} | steps_per_layer={{li: len(picked[li]) for li in picked}}")

    print(f"[DONE] saved={ok_cnt}, skipped={skip_cnt}, out_dir={STEERED_OUT_DIR}")

if __name__ == "__main__":
    main()
