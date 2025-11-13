import glob, os, yaml

opt = yaml.safe_load(open("opts/opts_se_hybrid.yml"))
roots = opt.get("train_roots", opt.get("train_root"))
if isinstance(roots, str):
    roots = [roots]

exts = ('.jpg','.jpeg','.png','.bmp','.tiff')
for r in roots:
    files = [p for p in glob.glob(os.path.join(r, "**", "*.*"), recursive=True) if p.lower().endswith(exts)]
    print(r, " -> found", len(files), "images")
    print("First few:", files[:5])
