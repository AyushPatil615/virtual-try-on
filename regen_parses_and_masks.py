from PIL import Image
import os, glob

# person images -> parse (single-channel zeros)
img_dir = r"my_data/test/image"
parse_dir = r"my_data/test/image-parse-v3"
os.makedirs(parse_dir, exist_ok=True)

for f in glob.glob(os.path.join(img_dir, "*")):
    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    base = os.path.splitext(os.path.basename(f))[0]
    out_path = os.path.join(parse_dir, base + ".png")
    with Image.open(f) as im:
        w,h = im.size
    # create single-channel 'L' image filled with 0 (background) sized to the person image
    Image.new("L", (w,h), 0).save(out_path)

print("Created/updated parses in:", parse_dir)

# cloth-warp masks -> white masks sized like cloth-warp images
warp_dir = r"my_data/test/cloth-warp"
warp_mask_dir = r"my_data/test/cloth-warp-mask"
os.makedirs(warp_mask_dir, exist_ok=True)
for f in glob.glob(os.path.join(warp_dir, "*")):
    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    base = os.path.splitext(os.path.basename(f))[0]
    out_path = os.path.join(warp_mask_dir, base + ".png")
    with Image.open(f) as im:
        w,h = im.size
    # white mask (255) indicating cloth region
    Image.new("L", (w,h), 255).save(out_path)

print("Created/updated cloth-warp masks in:", warp_mask_dir)

# unpaired-cloth-warp masks -> white masks sized like unpaired cloth warp images
unwarp_dir = r"my_data/test/unpaired-cloth-warp"
unwarp_mask_dir = r"my_data/test/unpaired-cloth-warp-mask"
os.makedirs(unwarp_mask_dir, exist_ok=True)
for f in glob.glob(os.path.join(unwarp_dir, "*")):
    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    base = os.path.splitext(os.path.basename(f))[0]
    out_path = os.path.join(unwarp_mask_dir, base + ".png")
    with Image.open(f) as im:
        w,h = im.size
    Image.new("L", (w,h), 255).save(out_path)

print("Created/updated unpaired cloth-warp masks in:", unwarp_mask_dir)
