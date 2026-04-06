import glob, binascii, sys, os
files = glob.glob("configs/**/*.yaml", recursive=True)
if not files:
    print("No .yaml files found under configs/")
    sys.exit(0)
for f in sorted(files):
    try:
        with open(f,"rb") as fh:
            b = fh.read(16)
    except Exception as e:
        print(f"ERROR reading {f}: {e}")
        continue
    print(f)
    print("PATH:", f)
    print("  first 16 bytes (hex):", binascii.hexlify(b).decode("ascii", errors="replace"))
    try:
        sample = b + (open(f,"rb").read(200) if len(b) < 200 else b"")
        txt = sample.decode("utf-8")
        sample_text = txt.splitlines()[:5]
        print("  looks like UTF-8 text - sample first lines:")
        for line in sample_text:
            print("    ", line)
    except Exception:
        print("  NOT UTF-8 (likely binary or different encoding)")
