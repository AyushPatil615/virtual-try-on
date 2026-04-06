# coding=utf-8
"""
DCI-VTON PREPROCESSING PIPELINE (SegFormer + torso-aware warp)

REQUIRES (in your dci-vton env):
    pip install torch torchvision torchaudio
    pip install transformers mediapipe pillow opencv-python accelerate safetensors

OUTPUT STRUCTURE (under DATASET_ROOT):
datasets/custom/
├── test_pairs.txt
└── test/
    ├── image/person.jpg
    ├── cloth/cloth.jpg
    ├── cloth-mask/cloth.jpg
    ├── image-parse-v3/person.png
    ├── image-parse-agnostic-v3.2/person.png
    ├── openpose_json/person_keypoints.json
    ├── openpose_img/person_rendered.png
    ├── image-densepose/person.jpg
    ├── cloth-warp/cloth.png
    ├── cloth-warp-mask/cloth.png
    ├── unpaired-cloth-warp/cloth.png
    └── unpaired-cloth-warp-mask/cloth.png
"""

import os
import cv2
import json
import torch
import numpy as np
import mediapipe as mp
from PIL import Image

# ----------------------------------------------------------------------
# CONFIG – CHANGE THESE THREE PATHS
# ----------------------------------------------------------------------
DATASET_ROOT = r"C:\Users\ADMIN\OneDrive\Desktop\dci_vton\DCI-VTON-Virtual-Try-On\datasets\custom"
PERSON_IMAGE = r"C:\Users\ADMIN\OneDrive\Desktop\dci_vton\DCI-VTON-Virtual-Try-On\input\person.jpg"
CLOTH_IMAGE  = r"C:\Users\ADMIN\OneDrive\Desktop\dci_vton\DCI-VTON-Virtual-Try-On\input\cloth.jpg"

PERSON_ID = "person"   # will save as person.jpg / person.png
CLOTH_ID  = "cloth"    # will save as cloth.jpg / cloth.png

TARGET_SIZE = (768, 1024)  # (W, H) used in DCI-VTON

# ----------------------------------------------------------------------
# UTILS
# ----------------------------------------------------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def resize_image(img, target_size=TARGET_SIZE):
    return cv2.resize(img, target_size)

# ----------------------------------------------------------------------
# 1. HUMAN PARSING (SegFormer only)
# ----------------------------------------------------------------------
def generate_human_parse(image_path, out_path):
    """
    Use SegFormer (mattmdjaga/segformer_b2_clothes) for human parsing.
    If this fails, we raise an error instead of drawing stick figures.
    """
    from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

    print("[1/11] Human parsing (SegFormer)...")
    model_name = "mattmdjaga/segformer_b2_clothes"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
    model.eval()

    img_pil = Image.open(image_path).convert("RGB")
    inputs = processor(images=img_pil, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    upsampled = torch.nn.functional.interpolate(
        logits,
        size=img_pil.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred_seg = upsampled.argmax(dim=1)[0].cpu().numpy()

    atr_map = np.zeros_like(pred_seg, dtype=np.uint8)
    # SegFormer label → ATR label mapping
    mapping = {
        0: 0,   # background
        1: 1,   # hat
        2: 2,   # hair
        3: 4,   # upper body (coat)
        4: 5,   # upper clothes
        5: 12,  # skirt
        6: 9,   # pants
        7: 6,   # dress
        8: 8,   # jumpsuit
        9: 18,  # left shoe
        10: 19, # right shoe
        11: 13, # face
        12: 16, # left leg
        13: 17, # right leg
        14: 14, # left arm
        15: 15, # right arm
    }
    for seg_label, atr_label in mapping.items():
        atr_map[pred_seg == seg_label] = atr_label

    ensure_dir(os.path.dirname(out_path))
    Image.fromarray(atr_map).save(out_path)
    print(f"       ✔ {out_path} (SegFormer)")
    return atr_map

# ----------------------------------------------------------------------
# 2. AGNOSTIC PARSING
# ----------------------------------------------------------------------
def generate_agnostic_parse(parse_path, out_path):
    """
    Make an agnostic parse by keeping only:
      - 0: background
      - 13: face
      - 14: left arm
      - 15: right arm
      - 16: left leg
      - 17: right leg
    and zeroing everything else (all clothing etc.).
    """
    print("[2/11] Agnostic parsing...")
    parse = np.array(Image.open(parse_path))
    agnostic = parse.copy()

    # labels we want to keep (body only)
    keep = {0, 13, 14, 15, 16, 17}

    unique_labels = np.unique(parse)
    # OPTIONAL: print to debug
    print("       parse unique labels:", unique_labels)

    for label in unique_labels:
        if label not in keep:
            agnostic[parse == label] = 0

    ensure_dir(os.path.dirname(out_path))
    Image.fromarray(agnostic.astype(np.uint8)).save(out_path)
    print(f"       ✔ {out_path}")

# ----------------------------------------------------------------------
# 3. OPENPOSE KEYPOINTS (JSON) via MediaPipe
# ----------------------------------------------------------------------
def generate_openpose_json(image_path, out_path, person_id):
    print("[3/11] OpenPose keypoints JSON (from MediaPipe)...")
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    keypoints = [0.0] * (18 * 3)

    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            model_complexity=1,
        )
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        # Map from MediaPipe indices to OpenPose 18-key format
        mp_to_op = {
            0: 0,
            2: 16,
            5: 15,
            11: 5,
            12: 2,
            13: 6,
            14: 3,
            15: 7,
            16: 4,
            23: 11,
            24: 8,
            25: 12,
            26: 9,
            27: 13,
            28: 10,
        }

        if result.pose_landmarks:
            for mp_idx, op_idx in mp_to_op.items():
                if mp_idx < len(result.pose_landmarks.landmark):
                    lm = result.pose_landmarks.landmark[mp_idx]
                    keypoints[op_idx * 3]     = float(lm.x * w)
                    keypoints[op_idx * 3 + 1] = float(lm.y * h)
                    keypoints[op_idx * 3 + 2] = float(lm.visibility)

            # neck (1) as midpoint of shoulders (2 & 5 in OP indexing)
            if keypoints[5 * 3 + 2] > 0 and keypoints[2 * 3 + 2] > 0:
                keypoints[1 * 3]     = (keypoints[5 * 3] + keypoints[2 * 3]) / 2
                keypoints[1 * 3 + 1] = (keypoints[5 * 3 + 1] + keypoints[2 * 3 + 1]) / 2
                keypoints[1 * 3 + 2] = min(keypoints[5 * 3 + 2], keypoints[2 * 3 + 2])

        pose.close()
    except Exception as e:
        print(f"       ⚠ Pose detection warning: {e}")

    data = {
        "version": 1.3,
        "people": [{
            "person_id": [person_id],
            "pose_keypoints_2d": keypoints,
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": [],
        }]
    }

    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"       ✔ {out_path}")
    return keypoints, (h, w)

# ----------------------------------------------------------------------
# 4. OPENPOSE RENDERED SKELETON
# ----------------------------------------------------------------------
def generate_openpose_rendered(keypoints, img_size, out_path):
    print("[4/11] OpenPose skeleton PNG...")
    h, w = img_size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    connections = [
        (1, 0, (255, 0, 0)), (1, 2, (255, 85, 0)), (2, 3, (255, 170, 0)),
        (3, 4, (255, 255, 0)), (1, 5, (170, 255, 0)), (5, 6, (85, 255, 0)),
        (6, 7, (0, 255, 0)), (1, 8, (0, 255, 85)), (8, 9, (0, 255, 170)),
        (9, 10, (0, 255, 255)), (1, 11, (0, 170, 255)), (11, 12, (0, 85, 255)),
        (12, 13, (0, 0, 255)), (0, 15, (255, 0, 170)), (0, 16, (170, 0, 255)),
    ]

    for start, end, color in connections:
        x1, y1, c1 = int(keypoints[start * 3]), int(keypoints[start * 3 + 1]), keypoints[start * 3 + 2]
        x2, y2, c2 = int(keypoints[end * 3]), int(keypoints[end * 3 + 1]), keypoints[end * 3 + 2]
        if c1 > 0.1 and c2 > 0.1:
            cv2.line(canvas, (x1, y1), (x2, y2), color, 4)

    for i in range(18):
        x, y, c = int(keypoints[i * 3]), int(keypoints[i * 3 + 1]), keypoints[i * 3 + 2]
        if c > 0.1:
            cv2.circle(canvas, (x, y), 6, (255, 255, 255), -1)
            cv2.circle(canvas, (x, y), 4, (0, 0, 0), -1)

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, canvas)
    print(f"       ✔ {out_path}")

# ----------------------------------------------------------------------
# 5. CLOTH MASK
# ----------------------------------------------------------------------
def generate_cloth_mask(cloth_path, out_path):
    print("[5/11] Cloth mask...")
    cloth = cv2.imread(cloth_path)
    gray = cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)

    # Assuming light background; adjust threshold if needed
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, -1)
        mask = filled

    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, mask)
    print(f"       ✔ {out_path}")

# ----------------------------------------------------------------------
# 6. DENSEPOSE (simple approximation)
# ----------------------------------------------------------------------
def generate_densepose(image_path, out_path):
    print("[6/11] DensePose IUV approximation...")
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    I_ch = np.zeros((h, w), np.uint8)
    U_ch = np.zeros((h, w), np.uint8)
    V_ch = np.zeros((h, w), np.uint8)

    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    rect = (int(w * 0.1), int(h * 0.02), int(w * 0.8), int(h * 0.95))
    cv2.grabCut(img, mask, rect, bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
    fg = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")

    parts = [
        (0, int(h * 0.22), int(w * 0.28), int(w * 0.72), 23),
        (int(h * 0.22), int(h * 0.62), int(w * 0.28), int(w * 0.72), 1),
        (int(h * 0.30), int(h * 0.48), int(w * 0.08), int(w * 0.28), 19),
        (int(h * 0.48), int(h * 0.62), int(w * 0.05), int(w * 0.28), 20),
        (int(h * 0.30), int(h * 0.48), int(w * 0.72), int(w * 0.92), 21),
        (int(h * 0.48), int(h * 0.62), int(w * 0.72), int(w * 0.95), 22),
        (int(h * 0.62), int(h * 0.80), int(w * 0.30), int(w * 0.50), 13),
        (int(h * 0.80), int(h * 0.95), int(w * 0.32), int(w * 0.50), 14),
        (int(h * 0.62), int(h * 0.80), int(w * 0.50), int(w * 0.70), 11),
        (int(h * 0.80), int(h * 0.95), int(w * 0.50), int(w * 0.68), 12),
    ]

    for y1, y2, x1, x2, part_id in parts:
        region = (fg[y1:y2, x1:x2] > 0)
        I_ch[y1:y2, x1:x2][region] = part_id

    for part_id in np.unique(I_ch):
        if part_id == 0:
            continue
        part_mask = (I_ch == part_id)
        ys, xs = np.where(part_mask)
        if len(xs) == 0:
            continue
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        for y, x in zip(ys, xs):
            U_ch[y, x] = int(((x - x_min) / (x_max - x_min)) * 255) if x_max > x_min else 128
            V_ch[y, x] = int(((y - y_min) / (y_max - y_min)) * 255) if y_max > y_min else 128

    iuv = cv2.merge([I_ch, U_ch, V_ch])
    ensure_dir(os.path.dirname(out_path))
    cv2.imwrite(out_path, iuv)
    print(f"       ✔ {out_path}")

# ----------------------------------------------------------------------
# 7. CLOTH WARPING to UPPER BODY
# ----------------------------------------------------------------------
def warp_cloth_to_upper_body(cloth_path, parse_path, warp_out, mask_out):
    """
    Robust static warp (simple + stable):

    - Build upper-body mask: upper clothes (4,5,6,7) + top of arms (14,15).
    - Compute bounding box of that region.
    - Scale cloth to FIT inside this box (preserve aspect ratio).
    - Center scaled cloth inside the box (no out-of-bounds).
    - Use upper-body mask to carve final cloth silhouette.
    - Lightly shrink + blur mask edges.
    """
    print("       [warp] cloth → upper-body region (robust warp)")

    try:
        # ------------------ 1) Read parse & build upper-body mask ------------------
        parse = cv2.imread(parse_path, cv2.IMREAD_GRAYSCALE)
        if parse is None:
            raise RuntimeError(f"Cannot read parse map: {parse_path}")
        H, W = parse.shape[:2]

        # labels: 5 upper clothes, 4/6/7 coat/dress, 14/15 arms
        upper = (parse == 5) | (parse == 4) | (parse == 6) | (parse == 7)
        arms = (parse == 14) | (parse == 15)

        ys, xs = np.where(arms)
        if len(xs) > 0:
            # top ~35% of arm pixels → sleeves
            y_cut = int(np.percentile(ys, 35))
            arms_top = np.zeros_like(arms, dtype=bool)
            arms_top[0:y_cut, :] = arms[0:y_cut, :]
            upper = upper | arms_top

        upper_mask = upper.astype(np.uint8)
        ys_u, xs_u = np.where(upper_mask == 1)
        if len(xs_u) == 0 or len(ys_u) == 0:
            raise RuntimeError("No upper-body region found in parse map.")

        x_min, x_max = xs_u.min(), xs_u.max()
        y_min, y_max = ys_u.min(), ys_u.max()

        # crop vertically so it doesn't go too low
        h_box = y_max - y_min + 1
        y_max = y_min + int(h_box * 0.85)

        target_w = x_max - x_min + 1
        target_h = y_max - y_min + 1

        print(f"       [warp] parse shape = {H}x{W}, bbox = {target_w}x{target_h}")

        # ------------------ 2) Read cloth & build cloth mask ------------------
        cloth = cv2.imread(cloth_path)
        if cloth is None:
            raise RuntimeError(f"Cannot read cloth image: {cloth_path}")
        ch, cw = cloth.shape[:2]
        print(f"       [warp] cloth shape = {ch}x{cw}")

        gray = cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)
        _, c_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Tight crop to remove background border
        contours, _ = cv2.findContours(c_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cloth = cloth[y_c:y_c + h_c, x_c:x_c + w_c]
            c_mask = c_mask[y_c:y_c + h_c, x_c:x_c + w_c]
            ch, cw = cloth.shape[:2]
            print(f"       [warp] cropped cloth shape = {ch}x{cw}")

        # ------------------ 3) Scale cloth to FIT inside target box ------------------
        # Use min(...) so new cloth is smaller than or equal to target box
        scale = min(target_w / cw, target_h / ch)
        new_w = max(1, int(cw * scale))
        new_h = max(1, int(ch * scale))
        cloth_scaled = cv2.resize(cloth, (new_w, new_h))
        c_mask_scaled = cv2.resize(c_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        print(f"       [warp] scaled cloth = {new_h}x{new_w}, scale={scale:.3f}")

        # ------------------ 4) Place scaled cloth centered in a target_h x target_w box ------------------
        cloth_box = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
        mask_box = np.zeros((target_h, target_w), dtype=np.uint8)

        x_off = (target_w - new_w) // 2
        y_off = (target_h - new_h) // 2
        x_off = max(0, x_off)
        y_off = max(0, y_off)

        cloth_box[y_off:y_off + new_h, x_off:x_off + new_w] = cloth_scaled
        mask_box[y_off:y_off + new_h, x_off:x_off + new_w] = c_mask_scaled

        # ------------------ 5) Apply upper-body mask (same size) ------------------
        upper_crop = upper_mask[y_min:y_max + 1, x_min:x_max + 1]  # target_h x target_w

        applied_mask = cv2.bitwise_and(
            mask_box,
            mask_box,
            mask=upper_crop.astype(np.uint8) * 255
        )

        # ------------------ 6) Light shrink + feather (simple, stable) ------------------
        kernel_erode = np.ones((2, 2), np.uint8)
        applied_mask = cv2.erode(applied_mask, kernel_erode, iterations=1)
        applied_mask = cv2.GaussianBlur(applied_mask, (3, 3), 0)

        # ------------------ 7) Paste into full-size canvases ------------------
        warp_canvas = np.full((H, W, 3), 255, dtype=np.uint8)
        mask_canvas = np.zeros((H, W), dtype=np.uint8)

        roi = warp_canvas[y_min:y_max + 1, x_min:x_max + 1]
        roi_mask = mask_canvas[y_min:y_max + 1, x_min:x_max + 1]

        cloth_fg = cv2.bitwise_and(cloth_box, cloth_box, mask=applied_mask)
        bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(applied_mask))
        roi_result = cv2.add(bg, cloth_fg)

        roi[:] = roi_result
        roi_mask[:] = applied_mask

        # ------------------ 8) Save ------------------
        ensure_dir(os.path.dirname(warp_out))
        ensure_dir(os.path.dirname(mask_out))
        ok1 = cv2.imwrite(warp_out, warp_canvas)
        ok2 = cv2.imwrite(mask_out, mask_canvas)
        print(f"       [warp] imwrite warp={ok1}, mask={ok2}")
        print(f"       ✔ {warp_out}")
        print(f"       ✔ {mask_out}")

    except Exception as e:
        print(f"       ✘ warp error: {e}")







# ----------------------------------------------------------------------
# 8. TEST PAIRS FILE
# ----------------------------------------------------------------------
def generate_test_pairs(out_path, person_id, cloth_id):
    print("[10/11] Test pairs file...")
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        f.write(f"{person_id}.jpg {cloth_id}.jpg\n")
    print(f"       ✔ {out_path}")

# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DCI-VTON COMPLETE PREPROCESSING PIPELINE (SegFormer + torso warp)")
    print("=" * 70)

    if not os.path.exists(PERSON_IMAGE):
        print(f"✘ ERROR: Person image not found: {PERSON_IMAGE}")
        return
    if not os.path.exists(CLOTH_IMAGE):
        print(f"✘ ERROR: Cloth image not found: {CLOTH_IMAGE}")
        return

    ensure_dir(DATASET_ROOT)
    TEST_DIR = os.path.join(DATASET_ROOT, "test")
    ensure_dir(TEST_DIR)

    # [0/11] copy & resize inputs
    print("\n[0/11] Preparing input images...")
    image_dir = os.path.join(TEST_DIR, "image")
    cloth_dir = os.path.join(TEST_DIR, "cloth")
    ensure_dir(image_dir)
    ensure_dir(cloth_dir)

    person_target = os.path.join(image_dir, f"{PERSON_ID}.jpg")
    cloth_target  = os.path.join(cloth_dir,  f"{CLOTH_ID}.jpg")

    person_img = resize_image(cv2.imread(PERSON_IMAGE), TARGET_SIZE)
    cloth_img  = resize_image(cv2.imread(CLOTH_IMAGE),  TARGET_SIZE)

    cv2.imwrite(person_target, person_img)
    cv2.imwrite(cloth_target,  cloth_img)
    print(f"       ✔ {person_target}")
    print(f"       ✔ {cloth_target}")

    # [1/11] Human parse
    print()
    parse_out = os.path.join(TEST_DIR, "image-parse-v3", f"{PERSON_ID}.png")
    generate_human_parse(person_target, parse_out)

    # [2/11] Agnostic parse
    print()
    agnostic_out = os.path.join(TEST_DIR, "image-parse-agnostic-v3.2", f"{PERSON_ID}.png")
    generate_agnostic_parse(parse_out, agnostic_out)

    # [3/11] OpenPose JSON
    print()
    pose_json = os.path.join(TEST_DIR, "openpose_json", f"{PERSON_ID}_keypoints.json")
    keypoints, img_size = generate_openpose_json(person_target, pose_json, PERSON_ID)

    # [4/11] OpenPose rendered
    print()
    pose_img = os.path.join(TEST_DIR, "openpose_img", f"{PERSON_ID}_rendered.png")
    generate_openpose_rendered(keypoints, img_size, pose_img)

    # [5/11] Cloth mask
    print()
    cloth_mask = os.path.join(TEST_DIR, "cloth-mask", f"{CLOTH_ID}.jpg")
    generate_cloth_mask(cloth_target, cloth_mask)

    # [6/11] DensePose
    print()
    densepose = os.path.join(TEST_DIR, "image-densepose", f"{PERSON_ID}.jpg")
    generate_densepose(person_target, densepose)

    # [7/11] Cloth warp (paired)
    print()
    print("[7/11] Cloth warp (paired)...")
    warp = os.path.join(TEST_DIR, "cloth-warp", f"{CLOTH_ID}.png")
    warp_mask = os.path.join(TEST_DIR, "cloth-warp-mask", f"{CLOTH_ID}.png")
    warp_cloth_to_upper_body(cloth_target, parse_out, warp, warp_mask)

    # [8/11] Unpaired cloth warp (reuse same method)
    print()
    print("[8/11] Unpaired cloth warp...")
    unwarp = os.path.join(TEST_DIR, "unpaired-cloth-warp", f"{CLOTH_ID}.png")
    unwarp_mask = os.path.join(TEST_DIR, "unpaired-cloth-warp-mask", f"{CLOTH_ID}.png")
    warp_cloth_to_upper_body(cloth_target, parse_out, unwarp, unwarp_mask)

    # [10/11] Test pairs
    print()
    pairs_file = os.path.join(DATASET_ROOT, "test_pairs.txt")
    generate_test_pairs(pairs_file, PERSON_ID, CLOTH_ID)

    print("\n" + "=" * 70)
    print("✅ ALL COMPONENTS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nRun DCI-VTON inference with:")
    print(f"   python test.py --dataroot {DATASET_ROOT}")

if __name__ == "__main__":
    main()