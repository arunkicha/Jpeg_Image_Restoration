import os, sys
from PIL import Image

REF = r"C:\Vit\Fall semester 25-26\Project\SWE1010\Datasets\Validation Dataset\live1\refimgs"
OUT_ROOT = r"C:\Vit\Fall semester 25-26\Project\SWE1010\Datasets\Validation Dataset\live1\color"
QFS = [10,20,30,40,50,70]

def main():
    if not os.path.isdir(REF):
        print("Ref folder not found:", REF); sys.exit(1)
    os.makedirs(OUT_ROOT, exist_ok=True)
    names = [f for f in os.listdir(REF) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    print("Found", len(names), "ref images.")
    for q in QFS:
        outdir = os.path.join(OUT_ROOT, f"qf_{q}")
        os.makedirs(outdir, exist_ok=True)
        print(f"[QF={q}] -> {outdir}")
        for n in names:
            img = Image.open(os.path.join(REF, n)).convert("RGB")
            out_path = os.path.join(outdir, os.path.splitext(n)[0] + ".jpg")
            img.save(dst_path, format='JPEG', quality=QF, subsampling=0, optimize=True)
    # inside the loop where q is defined
    if q == 40:
        img.save(out_path, format="JPEG", quality=q, subsampling=2, optimize=False)
    else:
        img.save(out_path, format="JPEG", quality=q, subsampling=2, optimize=True)

if __name__ == "__main__":
    main()
