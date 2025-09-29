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
ALPHA                  = 1.0
DEBUG_HOOK             = True
RESAMPLE_DIFFS_TO_NFE  = True
ALLOW_DIM_PROJ         = True
FORCE_NOISE_TEST       = False

REMAIN_NUM       = 30  # {10, 30, 50}

# ★ CREMA-D path
CREMAD_AUDIO_ROOT = Path("CREMA-D audio path ")  # {spk}/{sid}.wav
CREMAD_META_JSONL = Path("CREMA-D meta(cremad_test.jsonl)")  # option

FORGET_ROOT       = Path("<forget CREMAD_test activation PATH>")
REMAIN_MEAN_DIR   = Path("<Remain mean activation PATH>")
DIFF_ROOT         = Path("<CREMAD_test diff activation PATH>")

# model/vocoder
VOCODER_NAME      = "vocos"
LOAD_VOC_LOCAL    = False
MODEL_CFG_YAML    = Path("<PATH for /configs/F5TTS_v1_Base.yaml>")
CKPT_FILE         = Path("/<PATH for /ckpts/F5TTS_v1_Base/model_1250000.safetensors>")
VOCAB_TXT         = ""

# filename pattern
LAYER_PAT_FORGET = re.compile(r"_layer_(\d+)\.npy$", re.IGNORECASE)
LAYER_PAT_REMAIN = re.compile(r"remain_\d+_mean_layer_(\d+)\.npy$", re.IGNORECASE)

# ======================= Ablation options =======================
# layer band mode: "lt_mu_minus_sigma" | "lt_mu" | "lt_mu_plus_sigma" | "all"
LAYER_BAND_MODE   = "lt_mu_plus_sigma"
LAYER_K           = 1.0
DROP_WORST        = False

# step selection rule: "lt_layer_mean" | "lt_layer_mu"
STEP_RULE         = "lt_layer_mean"

# output path
RUN_TAG_BASE      = ""
RUN_TAG           = f"{RUN_TAG_BASE}_r{REMAIN_NUM}_{ALPHA}"
STEERED_OUT_DIR   = Path("<Path for /results/{RUN_TAG}_{LAYER_BAND_MODE}")

SAVE_PICKED_JSON  = True
RUN_TAG           = f"{RUN_TAG}_{LAYER_BAND_MODE}_{STEP_RULE}"

# ======================= F5-TTS util =======================
from trus.infer.utils_infer import (
    infer_process, load_model, load_vocoder,
    target_rms as _TARGET_RMS, cross_fade_duration as _XF,
    nfe_step as _NFE_DEFAULT, cfg_strength as _CFG,
    sway_sampling_coef as _SWAY, speed as _SPEED, fix_duration as _FIXDUR
)

# ======================= text mapping =======================
TEXT_CODE2SENTENCE = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}

SID_RE = re.compile(r"^(?P<spk>\d{3,5})_(?P<code>[A-Z]{3})_(?P<emo>[A-Z]{3})_(?P<var>[A-Za-z0-9]+)$")

def parse_sid(sid: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    m = SID_RE.match(Path(sid).stem)
    if not m: return None, None, None
    return m.group("spk"), m.group("code"), m.group("emo")

def sid_to_wav(audio_root: Path, sid: str) -> Optional[str]:

    spk, _, _ = parse_sid(sid)
    if spk:
        p = audio_root / spk / f"{sid}.wav"
        if p.is_file(): return str(p)
    cand = list(audio_root.rglob(f"{sid}.wav"))
    return str(cand[0]) if cand else None

def load_text_map_from_jsonl(jsonl_path: Path) -> Dict[str, str]:
    """
    JSONL: {"id": "...", "text": "...", ...}
    return: {id -> text}
    """
    id2text: Dict[str, str] = {}
    try:
        if jsonl_path and jsonl_path.is_file():
            with jsonl_path.open("r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line: continue
                    obj = json.loads(line)
                    uid = str(obj.get("id") or "")
                    txt = (obj.get("text") or "").strip()
                    if uid and txt:
                        id2text[uid] = txt
    except Exception as e:
        print(f"[WARN] text jsonl parse fail: {e}")
    print(f"[INFO] id->text loaded from JSONL: {len(id2text)} entries")
    return id2text

# ======================= helper utils =======================
def _resolve_path(wav_path: str, audio_root: str | None):
    p = Path(wav_path)
    if p.is_file():
        return str(p)
    return str(Path(audio_root) / wav_path) if audio_root else wav_path

def _safe_duration_sec(path: Optional[str]):
    try:
        if not path: return None
        info = torchaudio.info(path)
        if info.num_frames and info.sample_rate:
            return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass
    return None

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

# ======================= Ablation option: selection rule for layer =======================
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
        mu = float(np.mean(cs)); sig = float(np.std(cs, ddof=0))
        thr = mu - sig; keep = np.where(cs < thr)[0]
    else:
        raise ValueError(f"Unknown STEP_RULE={rule}")
    return keep

# ======================= steerer : erase_proj =======================
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

# ======================= meta info of CREMA-D =======================
def read_cremad_from_forget(forget_root: Path, audio_root: Path,
                            id2text: Optional[Dict[str, str]] = None) -> List[Tuple[str,str,str,str,str]]:
    
    PROMPT_MIN, PROMPT_MAX, TOTAL_MAX = 1.0, 10.0, 40.0

    sids = [d.name for d in sorted(forget_root.iterdir()) if d.is_dir()]

    items: List[Dict] = []
    by_spk: Dict[str, List[Dict]] = {}

    def _text_for(sid: str) -> str:
        if id2text and sid in id2text:
            t = id2text[sid].strip()
            if t: return t
        _, code, _ = parse_sid(sid)
        return TEXT_CODE2SENTENCE.get(code or "", ".") if code else "."

    for sid in sids:
        spk, code, emo = parse_sid(sid)
        wav = sid_to_wav(audio_root, sid)
        dur = _safe_duration_sec(wav)
        if not (spk and wav and dur):
            print(f"[WARN] skip meta for sid={sid} (spk={spk}, wav={bool(wav)}, dur={dur})")
            continue
        text = _text_for(sid)
        items.append({"utt": sid, "spk": spk, "wav": wav, "dur": dur, "text": text})
        by_spk.setdefault(spk, []).append(items[-1])

    meta: List[Tuple[str,str,str,str,str]] = []
    miss_ref = 0

    for tgt in items:
        tgt_d = tgt["dur"]
        cands = [x for x in by_spk.get(tgt["spk"], [])
                 if x["utt"] != tgt["utt"] and x["dur"] and (PROMPT_MIN <= x["dur"] <= PROMPT_MAX)]
        if cands:
            prompt = min(cands, key=lambda x: abs((x["dur"] or PROMPT_MIN) - PROMPT_MIN))
        else:
            if (tgt_d >= PROMPT_MIN) and (2 * tgt_d <= TOTAL_MAX):
                prompt = tgt
            else:
                miss_ref += 1
                continue

        total = (prompt["dur"] or 0.0) + (tgt_d or 0.0)
        if total > TOTAL_MAX:
            continue

        prompt_text = prompt["text"]
        gen_text    = tgt["text"]
        meta.append((tgt["utt"], prompt_text, prompt["wav"], gen_text, tgt["wav"]))

    print(f"[INFO] CREMA-D pairing done: {len(meta)} samples (ref-miss {miss_ref})")
    return meta

# ======================= main =======================
def main():
    STEERED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    ema, voc = load_everything()
    dit, blocks = find_dit_and_blocks(ema)

    id2text = load_text_map_from_jsonl(CREMAD_META_JSONL) if CREMAD_META_JSONL.is_file() else {}

    meta = read_cremad_from_forget(FORGET_ROOT, CREMAD_AUDIO_ROOT, id2text=id2text)
    print(f"[INFO] CREMA-D meta loaded: {len(meta)} samples")

    ok_cnt, skip_cnt = 0, 0

    for (sid, prompt_text, prompt_wav, gt_text, gt_wav) in tqdm(meta, desc="[STEER-CREMAD-ABL]"):
        # (A) forget / remain_mean activation file 
        f_dir = FORGET_ROOT / sid
        if not f_dir.is_dir():
            print(f"[WARN] forget dir missing: {f_dir}; skip"); skip_cnt += 1; continue
        f_layers = map_layers(f_dir, LAYER_PAT_FORGET)
        r_layers = map_layers(REMAIN_MEAN_DIR, LAYER_PAT_REMAIN)
        common_layers = sorted(set(f_layers) & set(r_layers))
        if not common_layers:
            print(f"[WARN] no common layers for {sid}; skip"); skip_cnt += 1; continue

        # (B) calculate step-wise cos per layer 
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

        # (C) layer selection (4 modes of band)
        selected_layers, mu_layers, sigma_layers = pick_layers_by_band(
            layer_mean_cos, mode=LAYER_BAND_MODE, k_sigma=LAYER_K, drop_worst=DROP_WORST
        )
        if not selected_layers:
            print(f"[INFO] no layer selected by {LAYER_BAND_MODE} for {sid}; skip")
            skip_cnt += 1; continue

        # (D) step selection (2 mode)
        inject_steps_map: Dict[int, set] = {}
        for li in sorted(selected_layers):
            cs = layer_step_cos[li]
            keep_idx = pick_steps_for_layer(cs, rule=STEP_RULE)
            if keep_idx.size > 0:
                inject_steps_map[li] = set(int(x) for x in keep_idx.tolist())

        final_layers = [li for li in selected_layers if li in inject_steps_map and len(inject_steps_map[li]) > 0]
        if not final_layers:
            print(f"[INFO] no (layer,step) after step-rule {STEP_RULE} for {sid}; skip"); skip_cnt += 1; continue

        # (E) load diff 
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

        # (F) hook setting (inject erase_proj to FFN-out)
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
            inject_steps_map=inject_steps_map,
        )
        steerer.attach()

        # (G) synthesis
        ref_audio = _resolve_path(prompt_wav, None)
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

        # (H) save
        STEERED_OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_wav = STEERED_OUT_DIR / f"{sid}.wav"
        sf.write(str(out_wav), wav, sr)
        ok_cnt += 1

        # (I) save selection result JSON (option)
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
