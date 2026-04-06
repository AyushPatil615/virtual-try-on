"""
DCI-VTON Virtual Try-On — Production GUI App
=============================================
Single-file, single-click application.
Select a person photo and a garment photo → get a try-on result.

Run with:
    conda activate dci-vton
    python run_app.py
"""

# ── Fix invalid console handles when launched without a terminal ──────
# Must happen before ANY import that touches sys.stderr (tqdm, torch, etc.)
import io as _io
import sys as _sys_early

def _safe_stream(original):
    """Return original if it has a valid fileno(), else a StringIO dummy."""
    try:
        if original is not None:
            original.fileno()   # raises OSError if handle is invalid
        return original
    except (OSError, _io.UnsupportedOperation):
        return _io.StringIO()

_sys_early.stdout = _safe_stream(_sys_early.stdout)
_sys_early.stderr = _safe_stream(_sys_early.stderr)

# Also tell tqdm to never write to console (it checks this env var at import time)
import os as _os_early
_os_early.environ.setdefault("TQDM_DISABLE", "1")
# Tell PyTorch CUDA allocator to use expandable segments (fixes fragmentation
# OOM errors even when total free VRAM is large). Must be set before import torch.
_os_early.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STANDARD LIBRARY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import os, sys, json, shutil, subprocess, threading, queue, time, textwrap
from pathlib import Path
from typing import Optional, Tuple
from itertools import islice
from contextlib import nullcontext

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TKINTER (stdlib)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PATH SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "custom"
TEST_DIR     = DATASET_ROOT / "test"
OUTPUT_DIR   = PROJECT_ROOT / "outputs" / "custom_run" / "result"
PF_AFN_ROOT  = PROJECT_ROOT / "PF-AFN" / "PF-AFN_test"
CONFIG_PATH  = PROJECT_ROOT / "configs" / "viton512_v2.yaml"
CKPT_PATH    = PROJECT_ROOT / "checkpoints" / "viton512_v2.ckpt"
TARGET_SIZE  = (768, 1024)   # (W, H)

PERSON_ID = "person"
CLOTH_ID  = "cloth"

# ── Model caches (loaded once, reused across runs) ────────────────────
_SEGFORMER_CACHE = {"processor": None, "model": None}  # survives between requests
_VTON_MODEL_CACHE = {"model": None, "sampler": None, "device": None}  # DCI-VTON
DDIM_STEPS = 20   # balanced: faster than original 31, better quality than 15

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THEME
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BG       = "#0f0f1a"
BG2      = "#1a1a2e"
BG3      = "#16213e"
ACCENT   = "#6c63ff"
ACCENT2  = "#ff6584"
SUCCESS  = "#43e97b"
WARNING  = "#f6d743"
ERROR    = "#ff6b6b"
FG       = "#e0e0e0"
FG2      = "#9090b0"
CARD     = "#1e1e3a"
BORDER   = "#2a2a4a"
BTN_GRAD = "#6c63ff"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PIPELINE LOGIC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# ── Phase 1: Preprocessing ────────────────────────────────────────────

def phase1_preprocess(person_path: str, cloth_path: str, log):
    """Run all preprocessing steps. Returns True on success."""
    import cv2
    import numpy as np
    import mediapipe as mp

    log("━"*60)
    log("PHASE 1 — PREPROCESSING")
    log("━"*60)

    person_src = Path(person_path)
    cloth_src  = Path(cloth_path)

    # directories
    dirs = [
        TEST_DIR / "image",
        TEST_DIR / "cloth",
        TEST_DIR / "cloth-mask",
        TEST_DIR / "image-parse-v3",
        TEST_DIR / "image-parse-agnostic-v3.2",
        TEST_DIR / "openpose_json",
        TEST_DIR / "openpose_img",
        TEST_DIR / "image-densepose",
        TEST_DIR / "cloth-warp",
        TEST_DIR / "cloth-warp-mask",
        TEST_DIR / "unpaired-cloth-warp",
        TEST_DIR / "unpaired-cloth-warp-mask",
    ]
    for d in dirs:
        ensure_dir(d)

    # ── 0. Copy & resize inputs ───────────────────────────────────────
    log("\n[1/8] Copying & resizing input images...")
    person_target = str(TEST_DIR / "image"  / f"{PERSON_ID}.jpg")
    cloth_target  = str(TEST_DIR / "cloth"  / f"{CLOTH_ID}.jpg")

    p_img = cv2.imread(str(person_src))
    c_img = cv2.imread(str(cloth_src))
    if p_img is None:
        raise RuntimeError(f"Cannot read person image: {person_src}")
    if c_img is None:
        raise RuntimeError(f"Cannot read cloth image: {cloth_src}")

    p_img = cv2.resize(p_img, TARGET_SIZE)
    c_img = cv2.resize(c_img, TARGET_SIZE)
    cv2.imwrite(person_target, p_img)
    cv2.imwrite(cloth_target,  c_img)
    log(f"  ✓ person → {person_target}")
    log(f"  ✓ cloth  → {cloth_target}")

    # ── 1. Human parse (SegFormer) ────────────────────────────────────
    # CUDA_VISIBLE_DEVICES is NOT touched here - SegFormer already runs on
    # CPU because we never call model.cuda(). No CUDA context is created.
    log("\n[2/8] Human parsing (SegFormer b2_clothes — CPU)...")
    parse_out = str(TEST_DIR / "image-parse-v3" / f"{PERSON_ID}.png")
    atr_map = _generate_human_parse(person_target, parse_out, log)

    # ── 2. Agnostic parse ─────────────────────────────────────────────
    log("\n[3/8] Agnostic parse...")
    agnostic_out = str(TEST_DIR / "image-parse-agnostic-v3.2" / f"{PERSON_ID}.png")
    _generate_agnostic_parse(parse_out, agnostic_out, log)

    # ── 3. OpenPose JSON ──────────────────────────────────────────────
    log("\n[4/8] OpenPose keypoints (MediaPipe)...")
    pose_json = str(TEST_DIR / "openpose_json" / f"{PERSON_ID}_keypoints.json")
    keypoints, img_size = _generate_openpose_json(person_target, pose_json, log)

    # ── 4. OpenPose skeleton PNG ──────────────────────────────────────
    log("\n[5/8] OpenPose skeleton PNG...")
    pose_img = str(TEST_DIR / "openpose_img" / f"{PERSON_ID}_rendered.png")
    _generate_openpose_rendered(keypoints, img_size, pose_img, log)

    # ── 5. Cloth mask ─────────────────────────────────────────────────
    log("\n[6/8] Cloth mask...")
    cloth_mask_out = str(TEST_DIR / "cloth-mask" / f"{CLOTH_ID}.jpg")
    _generate_cloth_mask(cloth_target, cloth_mask_out, log)

    # ── 6. DensePose ─────────────────────────────────────────────────
    log("\n[7/8] DensePose approximation...")
    densepose_out = str(TEST_DIR / "image-densepose" / f"{PERSON_ID}.jpg")
    _generate_densepose(person_target, densepose_out, log)

    # ── 7. test_pairs.txt ────────────────────────────────────────────
    log("\n[8/8] Writing test_pairs.txt...")
    pairs_path = DATASET_ROOT / "test_pairs.txt"
    pairs_path.write_text(f"{PERSON_ID}.jpg {CLOTH_ID}.jpg\n", encoding="utf-8")
    log(f"  ✓ {pairs_path}")

    log("\n✅ Phase 1 complete!\n")
    return True


def _generate_human_parse(image_path, out_path, log):
    import torch, numpy as np
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    model_name = "mattmdjaga/segformer_b2_clothes"

    # ── Use cached model if already loaded (huge speedup on 2nd+ runs) ──
    if _SEGFORMER_CACHE["model"] is None:
        log(f"  Loading {model_name} (CPU) — first run only...")
        _SEGFORMER_CACHE["processor"] = AutoImageProcessor.from_pretrained(model_name)
        _SEGFORMER_CACHE["model"]     = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        _SEGFORMER_CACHE["model"].eval()
        log("  ✓ SegFormer loaded and cached for future runs")
    else:
        log("  ✓ SegFormer (using cached model — instant)")

    processor = _SEGFORMER_CACHE["processor"]
    model     = _SEGFORMER_CACHE["model"]

    img_pil = Image.open(image_path).convert("RGB")
    inputs = processor(images=img_pil, return_tensors="pt")  # CPU tensors
    with torch.no_grad():
        outputs = model(**inputs)
    upsampled = torch.nn.functional.interpolate(
        outputs.logits, size=img_pil.size[::-1], mode="bilinear", align_corners=False
    )
    pred_seg = upsampled.argmax(dim=1)[0].numpy()  # already on CPU

    mapping = {0:0,1:1,2:2,3:4,4:5,5:12,6:9,7:6,8:8,9:18,10:19,11:13,12:16,13:17,14:14,15:15}
    atr_map = np.zeros_like(pred_seg, dtype=np.uint8)
    for seg_lbl, atr_lbl in mapping.items():
        atr_map[pred_seg == seg_lbl] = atr_lbl

    ensure_dir(os.path.dirname(out_path))
    Image.fromarray(atr_map).save(out_path)
    log(f"  ✓ Saved parse: {out_path}")
    log("  ✓ SegFormer done (CPU-only, GPU untouched)")
    return atr_map


def _generate_agnostic_parse(parse_path, out_path, log):
    import numpy as np
    parse = np.array(Image.open(parse_path))
    agnostic = parse.copy()
    keep = {0, 13, 14, 15, 16, 17}  # background + face + arms + legs
    for label in np.unique(parse):
        if label not in keep:
            agnostic[parse == label] = 0
    ensure_dir(os.path.dirname(out_path))
    Image.fromarray(agnostic.astype(np.uint8)).save(out_path)
    log(f"  ✓ Saved agnostic parse: {out_path}")


def _generate_openpose_json(image_path, out_path, log):
    import cv2, mediapipe as mp, numpy as np
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    keypoints = [0.0] * (18 * 3)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    mp_to_op = {0:0,2:16,5:15,11:5,12:2,13:6,14:3,15:7,16:4,23:11,24:8,25:12,26:9,27:13,28:10}
    if result.pose_landmarks:
        for mp_idx, op_idx in mp_to_op.items():
            if mp_idx < len(result.pose_landmarks.landmark):
                lm = result.pose_landmarks.landmark[mp_idx]
                keypoints[op_idx*3]   = float(lm.x * w)
                keypoints[op_idx*3+1] = float(lm.y * h)
                keypoints[op_idx*3+2] = float(lm.visibility)
        if keypoints[5*3+2] > 0 and keypoints[2*3+2] > 0:
            keypoints[1*3]   = (keypoints[5*3]   + keypoints[2*3])   / 2
            keypoints[1*3+1] = (keypoints[5*3+1] + keypoints[2*3+1]) / 2
            keypoints[1*3+2] = min(keypoints[5*3+2], keypoints[2*3+2])
    pose.close()

    data = {"version":1.3,"people":[{"person_id":[PERSON_ID],
        "pose_keypoints_2d":keypoints,"face_keypoints_2d":[],"hand_left_keypoints_2d":[],
        "hand_right_keypoints_2d":[],"pose_keypoints_3d":[],"face_keypoints_3d":[],
        "hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}]}
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    log(f"  ✓ Saved OpenPose JSON: {out_path}")
    return keypoints, (h, w)


def _generate_openpose_rendered(keypoints, img_size, out_path, log):
    import cv2, numpy as np
    h, w = img_size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    connections = [
        (1,0,(255,0,0)),(1,2,(255,85,0)),(2,3,(255,170,0)),(3,4,(255,255,0)),
        (1,5,(170,255,0)),(5,6,(85,255,0)),(6,7,(0,255,0)),(1,8,(0,255,85)),
        (8,9,(0,255,170)),(9,10,(0,255,255)),(1,11,(0,170,255)),(11,12,(0,85,255)),
        (12,13,(0,0,255)),(0,15,(255,0,170)),(0,16,(170,0,255)),
    ]
    for start, end, color in connections:
        x1,y1,c1 = int(keypoints[start*3]),int(keypoints[start*3+1]),keypoints[start*3+2]
        x2,y2,c2 = int(keypoints[end*3]),  int(keypoints[end*3+1]),  keypoints[end*3+2]
        if c1 > 0.1 and c2 > 0.1:
            cv2.line(canvas, (x1,y1),(x2,y2),color,4)
    for i in range(18):
        x,y,c = int(keypoints[i*3]),int(keypoints[i*3+1]),keypoints[i*3+2]
        if c > 0.1:
            cv2.circle(canvas,(x,y),6,(255,255,255),-1)
            cv2.circle(canvas,(x,y),4,(0,0,0),-1)
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)
    log(f"  ✓ Saved skeleton: {out_path}")


def _generate_cloth_mask(cloth_path, out_path, log):
    import cv2, numpy as np
    cloth = cv2.imread(cloth_path)
    gray = cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)
    
    # Multi-threshold: handle both white and gray backgrounds
    _, mask_white = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Also grab near-white (240-255 luminance)
    hsv = cv2.cvtColor(cloth, cv2.COLOR_BGR2HSV)
    # Low-saturation + high-Value = background
    bg_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    fg_mask = cv2.bitwise_not(bg_mask)
    mask = cv2.bitwise_or(mask_white, fg_mask)
    mask = cv2.bitwise_and(mask, mask_white)
    mask = mask_white  # Use simple white-bg threshold — most reliable for flat-lay
    
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, -1)
        mask = filled

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, mask)
    log(f"  ✓ Saved cloth mask: {out_path}")


def _generate_densepose(image_path, out_path, log):
    import cv2, numpy as np
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    I_ch = np.zeros((h,w), np.uint8)
    U_ch = np.zeros((h,w), np.uint8)
    V_ch = np.zeros((h,w), np.uint8)
    mask = np.zeros((h,w), np.uint8)
    bgd,fgd = np.zeros((1,65),np.float64), np.zeros((1,65),np.float64)
    rect = (int(w*0.1),int(h*0.02),int(w*0.8),int(h*0.95))
    cv2.grabCut(img,mask,rect,bgd,fgd,8,cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask==2)|(mask==0),0,1).astype("uint8")
    parts = [
        (0,int(h*0.22),int(w*0.28),int(w*0.72),23),
        (int(h*0.22),int(h*0.62),int(w*0.28),int(w*0.72),1),
        (int(h*0.30),int(h*0.48),int(w*0.08),int(w*0.28),19),
        (int(h*0.48),int(h*0.62),int(w*0.05),int(w*0.28),20),
        (int(h*0.30),int(h*0.48),int(w*0.72),int(w*0.92),21),
        (int(h*0.48),int(h*0.62),int(w*0.72),int(w*0.95),22),
        (int(h*0.62),int(h*0.80),int(w*0.30),int(w*0.50),13),
        (int(h*0.80),int(h*0.95),int(w*0.32),int(w*0.50),14),
        (int(h*0.62),int(h*0.80),int(w*0.50),int(w*0.70),11),
        (int(h*0.80),int(h*0.95),int(w*0.50),int(w*0.68),12),
    ]
    for y1,y2,x1,x2,pid in parts:
        region = (fg[y1:y2,x1:x2]>0)
        I_ch[y1:y2,x1:x2][region] = pid
    for pid in np.unique(I_ch):
        if pid==0: continue
        pm = (I_ch==pid); ys,xs = np.where(pm)
        if len(xs)==0: continue
        xmn,xmx,ymn,ymx = xs.min(),xs.max(),ys.min(),ys.max()
        for y,x in zip(ys,xs):
            U_ch[y,x] = int(((x-xmn)/(xmx-xmn))*255) if xmx>xmn else 128
            V_ch[y,x] = int(((y-ymn)/(ymx-ymn))*255) if ymx>ymn else 128
    iuv = cv2.merge([I_ch,U_ch,V_ch])
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, iuv)
    log(f"  ✓ Saved DensePose: {out_path}")


# ── Phase 2: PF-AFN Warping ───────────────────────────────────────────

def phase2_warp(log) -> bool:
    """Run PF-AFN warping and apply improved post-processing."""
    import cv2, numpy as np, gc

    log("━"*60)
    log("PHASE 2 — PF-AFN WARPING")
    log("━"*60)

    pf_data    = PF_AFN_ROOT / "dataset"
    pf_results = PF_AFN_ROOT / "results"
    test_py    = PF_AFN_ROOT / "test.py"
    warp_ckpt  = PF_AFN_ROOT / "checkpoints" / "checkpoints" / "PFAFN" / "warp_model_final.pth"
    gen_ckpt   = PF_AFN_ROOT / "checkpoints" / "checkpoints" / "PFAFN" / "gen_model_final.pth"

    for p, label in [(PF_AFN_ROOT,"PF-AFN root"),(test_py,"test.py"),(warp_ckpt,"warp ckpt"),(gen_ckpt,"gen ckpt")]:
        if not Path(p).exists():
            raise RuntimeError(f"Missing {label}: {p}")

    # ── Copy data files ───────────────────────────────────────────────
    log("\n[1/5] Preparing PF-AFN data directory...")
    src_map = {
        "test_img":     (TEST_DIR / "image",      f"{PERSON_ID}.jpg"),
        "test_clothes": (TEST_DIR / "cloth",       f"{CLOTH_ID}.jpg"),
        "test_edge":    (TEST_DIR / "cloth-mask",  f"{CLOTH_ID}.jpg"),
    }
    for folder, (src_dir, fname) in src_map.items():
        dst = pf_data / folder
        if dst.exists(): shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_dir / fname, dst / fname)
        log(f"  ✓ {fname} → {folder}/")

    # ── Write demo.txt (CWD-relative when subprocess runs) ────────────
    log("\n[2/5] Writing demo.txt...")
    demo_txt = PF_AFN_ROOT / "demo.txt"
    demo_txt.write_text(f"{PERSON_ID}.jpg {CLOTH_ID}.jpg\n", encoding="utf-8")
    log(f"  ✓ {demo_txt}")

    # ── Clean old results ─────────────────────────────────────────────
    if pf_results.exists():
        shutil.rmtree(pf_results)

    # ── Run PF-AFN ────────────────────────────────────────────────────
    log("\n[3/5] Running PF-AFN warping model (this takes ~30-60s)...")
    cmd = [
        sys.executable, str(test_py),
        "--name", "demo",
        "--resize_or_crop", "None",
        "--batchSize", "1",
        "--gpu_ids", "0",
    ]

    # Pass PYTORCH_CUDA_ALLOC_CONF to fix fragmentation OOM.
    # Do NOT touch CUDA_VISIBLE_DEVICES — that causes 0xC0000005 crash.
    sub_env = os.environ.copy()
    sub_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sub_env["TQDM_DISABLE"]             = "1"
    sub_env["TOKENIZERS_PARALLELISM"]   = "false"

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
        cwd=str(PF_AFN_ROOT), env=sub_env,
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log(f"  [PF-AFN] {line}")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"PF-AFN failed with exit code {rc}")
    log("  ✓ PF-AFN complete")


    # ── Extract & clean warped cloth ──────────────────────────────────
    log("\n[4/5] Extracting warped cloth from composite...")
    out_dir  = pf_results / "demo" / "PFAFN"
    out_files = sorted(out_dir.glob("*.jpg"))
    if not out_files:
        out_files = sorted(out_dir.glob("*.png"))
    if not out_files:
        raise RuntimeError(f"No PF-AFN output found in {out_dir}")

    composite = Image.open(out_files[0]).convert("RGB")
    W, H = composite.size
    panel_w = W // 3
    # Layout: [person | warped_cloth | tryon]
    warped = composite.crop((panel_w, 0, panel_w * 2, H))
    if warped.size != TARGET_SIZE:
        warped = warped.resize(TARGET_SIZE, Image.LANCZOS)
    log(f"  ✓ Extracted warped cloth: {warped.size}")

    # ── Post-process: remove gray/white background ────────────────────
    log("\n[5/5] Cleaning warped cloth (multi-strategy BG removal)...")
    arr = np.array(warped).astype(np.float32)
    R,G,B = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    # A) pure white
    is_white = (R>235) & (G>235) & (B>235)
    # B) neutral gray (PF-AFN background fill ~182,182,182)
    rg = np.abs(R-G); rb = np.abs(R-B); gb = np.abs(G-B)
    bright = (R+G+B)/3.0
    is_gray = (rg<18) & (rb<18) & (gb<18) & (bright>90)
    raw_mask = np.where(is_white | is_gray, 0, 255).astype(np.uint8)
    log(f"  ✓ Background pixels: {(is_white|is_gray).mean()*100:.1f}%")

    # Constrain with dilated original cloth-mask
    orig_mask_path = TEST_DIR / "cloth-mask" / f"{CLOTH_ID}.jpg"
    if orig_mask_path.exists():
        orig = np.array(Image.open(orig_mask_path).convert("L"))
        if orig.shape != (TARGET_SIZE[1], TARGET_SIZE[0]):
            orig = np.array(Image.fromarray(orig).resize(TARGET_SIZE, Image.NEAREST))
        orig_bin = (orig > 127).astype(np.uint8) * 255
        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
        orig_dil = cv2.dilate(orig_bin, k_dil, iterations=1)
        combined = cv2.bitwise_and(raw_mask, orig_dil)
        log("  ✓ Applied cloth-mask constraint")
    else:
        combined = raw_mask

    # Morphological cleanup
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    clean = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    clean = cv2.erode(clean, k_erode, iterations=1)

    # Keep largest connected component
    n, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if n > 2:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        clean = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Apply: set background to white
    arr_u8 = arr.astype(np.uint8)
    arr_u8[clean == 0] = 255
    cleaned = Image.fromarray(arr_u8, "RGB")
    coverage = float(np.sum(clean == 255)) / clean.size * 100
    log(f"  ✓ Clean mask coverage: {coverage:.1f}%")

    # ── Save outputs ──────────────────────────────────────────────────
    warp_dir  = TEST_DIR / "cloth-warp";       warp_dir.mkdir(parents=True, exist_ok=True)
    wmask_dir = TEST_DIR / "cloth-warp-mask";  wmask_dir.mkdir(parents=True, exist_ok=True)
    unwarp_dir  = TEST_DIR / "unpaired-cloth-warp";      unwarp_dir.mkdir(parents=True, exist_ok=True)
    unwmask_dir = TEST_DIR / "unpaired-cloth-warp-mask"; unwmask_dir.mkdir(parents=True, exist_ok=True)

    cleaned.save(warp_dir / f"{CLOTH_ID}.png")
    cleaned.save(unwarp_dir / f"{CLOTH_ID}.png")
    mask_img = Image.fromarray(clean, "L")
    mask_img.save(wmask_dir  / f"{CLOTH_ID}.png")
    mask_img.save(unwmask_dir / f"{CLOTH_ID}.png")
    log(f"  ✓ Saved cloth-warp")
    log(f"  ✓ Saved cloth-warp-mask")
    log("\n✅ Phase 2 complete!\n")
    return True


# ── Phase 3: DCI-VTON Inference ───────────────────────────────────────

def phase3_inference(log) -> Optional[str]:
    """Run DCI-VTON diffusion inference. Returns output image path."""
    log("━"*60)
    log("PHASE 3 — DCI-VTON INFERENCE")
    log("━"*60)

    # HuggingFace env fix
    for k in ["HF_ENDPOINT","HF_HUB_ENDPOINT","TRANSFORMERS_OFFLINE","HF_HUB_OFFLINE"]:
        os.environ.pop(k, None)
    os.environ["HF_ENDPOINT"]     = "https://huggingface.co"
    os.environ["HF_HUB_ENDPOINT"] = "https://huggingface.co"

    import torch, numpy as np
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.transforms import Resize
    from einops import rearrange
    from pytorch_lightning import seed_everything
    from types import SimpleNamespace
    from ldm.data.cp_dataset import CPDataset
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.ddim import DDIMSampler
    try:
        from torch.cuda.amp import autocast
    except Exception:
        autocast = None

    log(f"\n[1/4] Loading config & model...")
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Config not found: {CONFIG_PATH}")
    if not CKPT_PATH.exists():
        raise RuntimeError(f"Checkpoint not found: {CKPT_PATH}")

    seed_everything(42)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    log(f"  Device: {device}")

    # ── Use cached DCI-VTON model if already loaded (saves ~45s per run) ──
    if _VTON_MODEL_CACHE["model"] is None:
        log("  Loading DCI-VTON checkpoint — first run only...")
        config = OmegaConf.load(str(CONFIG_PATH))
        pl_sd  = torch.load(str(CKPT_PATH), map_location="cpu")
        sd     = pl_sd.get("state_dict", pl_sd)
        model  = instantiate_from_config(config.model)
        model.load_state_dict(sd, strict=False)
        if use_cuda: model.cuda()
        model.eval()
        _VTON_MODEL_CACHE["model"]   = model
        _VTON_MODEL_CACHE["sampler"] = DDIMSampler(model)
        _VTON_MODEL_CACHE["device"]  = device
        log("  ✓ DCI-VTON model loaded and cached for future runs")
    else:
        log("  ✓ DCI-VTON (using cached model — instant)")
        model  = _VTON_MODEL_CACHE["model"]
        device = _VTON_MODEL_CACHE["device"]
    log("  ✓ Model ready")

    log("\n[2/4] Loading dataset...")
    H = W = 512
    dataset = CPDataset(str(DATASET_ROOT), H, mode="test", unpaired=False)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=use_cuda)
    sampler = _VTON_MODEL_CACHE["sampler"] or DDIMSampler(model)
    log(f"  ✓ Dataset size: {len(dataset)}")

    ensure_dir(OUTPUT_DIR)

    precision_ctx = autocast if (autocast is not None and use_cuda) else nullcontext

    log(f"\n[3/4] Running DDIM sampling ({DDIM_STEPS} steps)...")
    result_path = None
    with torch.no_grad():
        with precision_ctx():
            try:    ema_ctx = model.ema_scope()
            except: ema_ctx = nullcontext()
            with ema_ctx:
                for data in loader:
                    mask_tensor    = data["inpaint_mask"]
                    inpaint_image  = data["inpaint_image"]
                    ref_tensor     = data["ref_imgs"]
                    feat_tensor    = data["warp_feat"]
                    image_tensor   = data["GT"]
                    filenames      = data.get("file_name", None)

                    test_model_kwargs = {
                        "inpaint_mask":  mask_tensor.to(device),
                        "inpaint_image": inpaint_image.to(device),
                    }
                    feat_tensor = feat_tensor.to(device)
                    ref_tensor  = ref_tensor.to(device)

                    uc = None
                    c  = model.get_learned_conditioning(ref_tensor.to(torch.float16))
                    c  = model.proj_out(c)

                    z_inpaint = model.encode_first_stage(test_model_kwargs["inpaint_image"])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs["inpaint_image"] = z_inpaint
                    test_model_kwargs["inpaint_mask"]  = Resize([z_inpaint.shape[-2], z_inpaint.shape[-1]])(
                        test_model_kwargs["inpaint_mask"]
                    )

                    warp_feat = model.encode_first_stage(feat_tensor)
                    warp_feat = model.get_first_stage_encoding(warp_feat).detach()
                    ts        = torch.full((1,), 999, device=device, dtype=torch.long)
                    start_code = model.q_sample(warp_feat, ts)

                    shape = [4, H//8, W//8]
                    samples, _ = sampler.sample(
                        S=DDIM_STEPS, conditioning=c, batch_size=1, shape=shape,
                        verbose=False, unconditional_guidance_scale=1.0,
                        unconditional_conditioning=uc, eta=0.0,
                        x_T=start_code, test_model_kwargs=test_model_kwargs
                    )

                    x_dec = model.decode_first_stage(samples)
                    x_dec = torch.clamp((x_dec+1.0)/2.0, 0.0, 1.0).cpu().permute(0,2,3,1).numpy()
                    x_dec_t = torch.from_numpy(x_dec).permute(0,3,1,2)

                    x_src = torch.clamp((image_tensor+1.0)/2.0, 0.0, 1.0)
                    x_res = x_dec_t * (1 - mask_tensor) + mask_tensor * x_src

                    resize_fn = transforms.Resize((H, int(H/256*192)))
                    for i, x_sample in enumerate(x_res):
                        fname = filenames[i] if filenames else f"result_{i}"
                        save_x = resize_fn(x_sample)
                        save_x = 255.0 * rearrange(save_x.cpu().numpy(), "c h w -> h w c")
                        img = Image.fromarray(save_x.astype(np.uint8))
                        stem = fname[:-4] if (fname.endswith(".jpg") or fname.endswith(".png")) else fname
                        out_file = Path(OUTPUT_DIR) / f"{stem}.png"
                        img.save(out_file)
                        result_path = str(out_file)
                        log(f"  ✓ Saved result: {out_file}")

    log("\n✅ Phase 3 complete!\n")
    return result_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GUI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_placeholder(text: str, size=(220, 280), color=BG3) -> ImageTk.PhotoImage:
    img = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(img)
    # draw dashed border
    for x in range(0, size[0], 10):
        draw.line([(x, 0),(min(x+5,size[0]),0)], fill=BORDER, width=2)
        draw.line([(x, size[1]-2),(min(x+5,size[0]),size[1]-2)], fill=BORDER, width=2)
    for y in range(0, size[1], 10):
        draw.line([(0,y),(0,min(y+5,size[1]))], fill=BORDER, width=2)
        draw.line([(size[0]-2,y),(size[0]-2,min(y+5,size[1]))], fill=BORDER, width=2)
    # icon
    cx, cy = size[0]//2, size[1]//2 - 20
    draw.ellipse([cx-18, cy-18, cx+18, cy+18], outline=ACCENT, width=2)
    draw.line([cx, cy-10, cx, cy+10], fill=ACCENT, width=2)
    draw.line([cx-10, cy, cx+10, cy], fill=ACCENT, width=2)
    # text
    try:
        for line_i, t in enumerate(textwrap.wrap(text, 18)):
            tw = len(t) * 5
            draw.text((cx - tw//2, cy + 30 + line_i*16), t, fill=FG2)
    except Exception:
        pass
    return ImageTk.PhotoImage(img)


def fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    return img


class VTryOnApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("DCI-VTON Virtual Try-On")
        self.geometry("1100x820")
        self.minsize(900, 700)
        self.configure(bg=BG)
        self.resizable(True, True)

        self._person_path: Optional[str] = None
        self._cloth_path:  Optional[str] = None
        self._running   = False
        self._log_queue: queue.Queue = queue.Queue()
        self._result_path: Optional[str] = None

        self._person_tk = None   # keep references to prevent GC
        self._cloth_tk  = None
        self._result_tk = None

        self._build_ui()
        self._poll_log()

    # ────────────────────────────────────────────────────────────────
    # UI BUILD
    # ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ───────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG, pady=18)
        hdr.pack(fill="x", padx=24)

        tk.Label(hdr, text="✦  Virtual Try-On", font=("Helvetica", 26, "bold"),
                 bg=BG, fg=FG).pack(side="left")
        tk.Label(hdr, text="DCI-VTON  ×  PF-AFN", font=("Helvetica", 11),
                 bg=BG, fg=FG2).pack(side="left", padx=16, pady=6)

        # ── Status bar ───────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Select a person photo and a garment to begin.")
        status_bar = tk.Label(self, textvariable=self._status_var,
                              bg=BG2, fg=FG2, font=("Helvetica", 10),
                              anchor="w", padx=16, pady=6)
        status_bar.pack(fill="x", padx=24)

        # ── Main content ─────────────────────────────────────────────
        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True, padx=24, pady=8)

        # Left: person picker
        self._person_card = self._build_picker_card(
            content, "PERSON PHOTO", "Click to select person image",
            self._pick_person
        )
        self._person_card.pack(side="left", fill="y", padx=(0,8))

        # Center: controls
        ctrl = tk.Frame(content, bg=BG)
        ctrl.pack(side="left", fill="both", expand=True)

        self._build_run_button(ctrl)
        self._build_progress_section(ctrl)
        self._build_log_area(ctrl)

        # Right: cloth picker
        self._cloth_card = self._build_picker_card(
            content, "GARMENT", "Click to select cloth image",
            self._pick_cloth
        )
        self._cloth_card.pack(side="right", fill="y", padx=(8,0))

    def _build_picker_card(self, parent, title, hint, command):
        card = tk.Frame(parent, bg=CARD, bd=0, highlightthickness=1,
                        highlightbackground=BORDER, highlightcolor=ACCENT)
        card.configure(cursor="hand2")

        tk.Label(card, text=title, bg=CARD, fg=ACCENT,
                 font=("Helvetica", 10, "bold"), pady=8).pack()

        # Image preview area
        preview_frame = tk.Frame(card, bg=CARD, width=220, height=280)
        preview_frame.pack(padx=12, pady=4)
        preview_frame.pack_propagate(False)

        lbl = tk.Label(preview_frame, bg=BG3, cursor="hand2")
        lbl.place(relwidth=1, relheight=1)
        lbl.bind("<Button-1>", lambda e: command())

        ph = make_placeholder(hint)
        lbl.configure(image=ph)
        lbl._ph = ph  # prevent GC

        btn = tk.Button(card, text="Browse…", bg=BG2, fg=FG,
                        font=("Helvetica", 9), bd=0, padx=10, pady=4,
                        activebackground=ACCENT, activeforeground="white",
                        cursor="hand2", command=command)
        btn.pack(pady=8)

        # Store refs
        card._preview_lbl = lbl
        card._title = title
        return card

    def _build_run_button(self, parent):
        frame = tk.Frame(parent, bg=BG)
        frame.pack(pady=16)

        self._run_btn = tk.Button(
            frame,
            text="  ▶  RUN VIRTUAL TRY-ON  ",
            font=("Helvetica", 14, "bold"),
            bg=ACCENT, fg="white",
            activebackground="#5a52d5", activeforeground="white",
            bd=0, padx=24, pady=12,
            cursor="hand2",
            command=self._start_pipeline,
        )
        self._run_btn.pack()

        # Hover effect
        self._run_btn.bind("<Enter>", lambda e: self._run_btn.configure(bg="#5a52d5"))
        self._run_btn.bind("<Leave>", lambda e: self._run_btn.configure(bg=ACCENT if not self._running else "#444"))

    def _build_progress_section(self, parent):
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill="x", pady=(0,8))

        # Phase labels
        phases_frame = tk.Frame(frame, bg=BG)
        phases_frame.pack(fill="x")

        self._phase_labels = []
        phases = ["① Preprocess", "② Warp", "③ Inference"]
        for i, p in enumerate(phases):
            lbl = tk.Label(phases_frame, text=p, bg=BG, fg=FG2,
                           font=("Helvetica", 9))
            lbl.grid(row=0, column=i, padx=8, pady=2)
            self._phase_labels.append(lbl)

        # Progress bar
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Custom.Horizontal.TProgressbar",
                        troughcolor=BG2, background=ACCENT,
                        lightcolor=ACCENT, darkcolor=ACCENT,
                        bordercolor=BG, thickness=8)

        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            frame, variable=self._progress_var,
            maximum=100, length=400,
            style="Custom.Horizontal.TProgressbar"
        )
        self._progress_bar.pack(fill="x", padx=4, pady=4)

    def _build_log_area(self, parent):
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Pipeline Log", bg=BG, fg=FG2,
                 font=("Helvetica", 9, "bold"), anchor="w").pack(fill="x", padx=4)

        log_frame = tk.Frame(frame, bg=BG2, highlightthickness=1,
                             highlightbackground=BORDER)
        log_frame.pack(fill="both", expand=True, pady=4)

        self._log_text = tk.Text(
            log_frame,
            bg=BG2, fg=FG, insertbackground=FG,
            font=("Consolas", 8),
            wrap="word", bd=0,
            state="disabled",
            relief="flat",
        )
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True, padx=4, pady=4)

        # Tag colors
        self._log_text.tag_configure("ok",  foreground=SUCCESS)
        self._log_text.tag_configure("err", foreground=ERROR)
        self._log_text.tag_configure("hdr", foreground=ACCENT, font=("Consolas", 8, "bold"))
        self._log_text.tag_configure("warn", foreground=WARNING)

    # ────────────────────────────────────────────────────────────────
    # IMAGE PICKING
    # ────────────────────────────────────────────────────────────────
    def _pick_person(self):
        path = filedialog.askopenfilename(
            title="Select Person Photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All","*.*")]
        )
        if path:
            self._person_path = path
            self._update_preview(self._person_card, path)
            self._status("Person photo loaded ✓")

    def _pick_cloth(self):
        path = filedialog.askopenfilename(
            title="Select Garment Photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All","*.*")]
        )
        if path:
            self._cloth_path = path
            self._update_preview(self._cloth_card, path)
            self._status("Garment photo loaded ✓")

    def _update_preview(self, card, path: str):
        try:
            img = Image.open(path).convert("RGB")
            img = fit_image(img, 220, 280)
            tk_img = ImageTk.PhotoImage(img)
            card._preview_lbl.configure(image=tk_img)
            card._preview_lbl._img = tk_img   # prevent GC
        except Exception as e:
            self._log_msg(f"Preview error: {e}", "warn")

    # ────────────────────────────────────────────────────────────────
    # PIPELINE
    # ────────────────────────────────────────────────────────────────
    def _start_pipeline(self):
        if self._running:
            return
        if not self._person_path:
            messagebox.showwarning("Missing Input", "Please select a person photo first.")
            return
        if not self._cloth_path:
            messagebox.showwarning("Missing Input", "Please select a garment photo first.")
            return

        self._running = True
        self._run_btn.configure(text="  ⏳  RUNNING…  ", state="disabled", bg="#444")
        self._progress_var.set(0)
        self._clear_log()
        self._set_phase(None)

        thread = threading.Thread(target=self._pipeline_thread, daemon=True)
        thread.start()

    def _pipeline_thread(self):
        # ── Guard against invalid console handles (no terminal) ────────
        # tqdm, pytorch-lightning and other libs write to stderr/stdout;
        # when launched via Start-Process without a console the handles are
        # invalid and raise OSError: [WinError 6]. Redirect to safe dummies.
        import io, sys, os
        def _safe(s):
            try:
                if s is not None: s.fileno()
                return s
            except (OSError, io.UnsupportedOperation):
                return io.StringIO()
        sys.stdout = _safe(sys.stdout)
        sys.stderr = _safe(sys.stderr)
        os.environ["TQDM_DISABLE"] = "1"   # suppress tqdm console writes

        try:
            def log(msg):
                self._log_queue.put(("msg", msg))

            # Phase 1
            self._log_queue.put(("phase", 0))
            self._log_queue.put(("progress", 5))
            phase1_preprocess(self._person_path, self._cloth_path, log)
            # Free GPU memory before PF-AFN subprocess
            try:
                import torch, gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                log("  ✓ GPU cleared after Phase 1")
            except Exception:
                pass
            self._log_queue.put(("progress", 33))

            # Phase 2
            self._log_queue.put(("phase", 1))
            self._log_queue.put(("progress", 36))
            phase2_warp(log)
            self._log_queue.put(("progress", 66))

            # Phase 3
            self._log_queue.put(("phase", 2))
            self._log_queue.put(("progress", 70))
            result = phase3_inference(log)
            self._log_queue.put(("progress", 100))

            if result and Path(result).exists():
                self._log_queue.put(("done", result))
            else:
                self._log_queue.put(("error", "Inference complete but result file not found."))

        except Exception as exc:
            import traceback
            self._log_queue.put(("error", str(exc)))
            self._log_queue.put(("msg", traceback.format_exc()))

    def _poll_log(self):
        """Drain the queue and update UI. Runs in main thread every 100ms."""
        try:
            while True:
                item = self._log_queue.get_nowait()
                kind, val = item
                if kind == "msg":
                    self._log_msg(val)
                elif kind == "progress":
                    self._progress_var.set(val)
                elif kind == "phase":
                    self._set_phase(val)
                elif kind == "done":
                    self._on_success(val)
                elif kind == "error":
                    self._on_error(val)
        except queue.Empty:
            pass
        self.after(80, self._poll_log)

    # ────────────────────────────────────────────────────────────────
    # UI HELPERS
    # ────────────────────────────────────────────────────────────────
    def _log_msg(self, msg: str, tag: str = ""):
        self._log_text.configure(state="normal")
        if not tag:
            if msg.startswith("  ✓") or "complete" in msg.lower() or "✅" in msg:
                tag = "ok"
            elif "━" in msg or "PHASE" in msg:
                tag = "hdr"
            elif "✘" in msg or "ERROR" in msg or "error" in msg.lower() or "Traceback" in msg:
                tag = "err"
            elif "⚠" in msg or "warn" in msg.lower():
                tag = "warn"
        self._log_text.insert("end", msg + "\n", tag if tag else ())
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _clear_log(self):
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    def _status(self, msg: str):
        self._status_var.set(msg)

    def _set_phase(self, idx: Optional[int]):
        for i, lbl in enumerate(self._phase_labels):
            if idx is None:
                lbl.configure(fg=FG2, font=("Helvetica", 9))
            elif i < idx:
                lbl.configure(fg=SUCCESS, font=("Helvetica", 9))
            elif i == idx:
                lbl.configure(fg=ACCENT, font=("Helvetica", 9, "bold"))
            else:
                lbl.configure(fg=FG2, font=("Helvetica", 9))

    def _reset_controls(self):
        self._running = False
        self._run_btn.configure(text="  ▶  RUN VIRTUAL TRY-ON  ", state="normal", bg=ACCENT)

    def _on_success(self, result_path: str):
        self._set_phase(3)  # all done
        self._status("✅ Try-on complete! Showing result…")
        self._log_msg(f"\n✅ SUCCESS — {result_path}", "ok")
        self._reset_controls()
        self._show_result_window(result_path)

    def _on_error(self, msg: str):
        self._status(f"❌ Error: {msg[:80]}")
        self._log_msg(f"\n❌ FAILED: {msg}", "err")
        self._reset_controls()
        messagebox.showerror("Pipeline Error", msg[:300])

    # ────────────────────────────────────────────────────────────────
    # RESULT WINDOW
    # ────────────────────────────────────────────────────────────────
    def _show_result_window(self, result_path: str):
        win = tk.Toplevel(self)
        win.title("✦ Try-On Result")
        win.configure(bg=BG)
        win.geometry("860x620")
        win.grab_set()

        tk.Label(win, text="✦ Virtual Try-On Result", font=("Helvetica", 20, "bold"),
                 bg=BG, fg=FG, pady=16).pack()

        img_frame = tk.Frame(win, bg=BG)
        img_frame.pack(fill="both", expand=True, padx=24, pady=8)

        MAX_H = 480
        labels_titles = [
            (self._person_path, "Original Person"),
            (str(TEST_DIR / "cloth" / f"{CLOTH_ID}.jpg"), "Garment"),
            (result_path, "✦ Try-On Result"),
        ]
        for path, title in labels_titles:
            col = tk.Frame(img_frame, bg=CARD, bd=0, highlightthickness=1,
                           highlightbackground=BORDER)
            col.pack(side="left", fill="both", expand=True, padx=6)

            tk.Label(col, text=title, bg=CARD, fg=ACCENT if "✦" in title else FG2,
                     font=("Helvetica", 10, "bold"), pady=6).pack()
            try:
                img = Image.open(path).convert("RGB")
                img.thumbnail((240, MAX_H), Image.LANCZOS)
                tk_img = ImageTk.PhotoImage(img)
                lbl = tk.Label(col, image=tk_img, bg=CARD)
                lbl.image = tk_img   # prevent GC
                lbl.pack(pady=8, padx=8)
            except Exception as e:
                tk.Label(col, text=f"Error: {e}", bg=CARD, fg=ERROR,
                         wraplength=200).pack()

        # Buttons
        btns = tk.Frame(win, bg=BG)
        btns.pack(pady=12)

        def save_result():
            dst = filedialog.asksaveasfilename(
                title="Save Result",
                defaultextension=".png",
                filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("All","*.*")],
                initialfile="tryon_result.png",
            )
            if dst:
                shutil.copy2(result_path, dst)
                messagebox.showinfo("Saved", f"Result saved to:\n{dst}")

        tk.Button(btns, text="💾  Save Result", bg=ACCENT, fg="white",
                  font=("Helvetica", 11, "bold"), bd=0, padx=20, pady=8,
                  cursor="hand2", command=save_result).pack(side="left", padx=8)

        tk.Button(btns, text="✕  Close", bg=BG2, fg=FG,
                  font=("Helvetica", 11), bd=0, padx=20, pady=8,
                  cursor="hand2", command=win.destroy).pack(side="left", padx=8)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    app = VTryOnApp()
    app.mainloop()
