# -*- coding: utf-8 -*-
"""
build_print_cards.py  --  Team SafePath | RVCE Hackathon 2026
=============================================================
Picks the sharpest sample image per defect class from each material
dataset and builds demo/print_cards/ with labeled JPEGs + a
print-ready HTML sheet.

Strategy per material:
  Steel  -- filename prefix (crazing_*, rolled-in_scale_*, ...)
  Aluminum/Wood -- label-file lookup (images have hash/serial names;
                   we read the .txt label to find which class each image is)

Usage:
    python tools/build_print_cards.py
"""

import os, sys, shutil, cv2, yaml, random
import numpy as np
from pathlib import Path

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = Path("demo/print_cards")
CARD_W, CARD_H = 420, 420
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ── Material configs ──────────────────────────────────────────────────────────
MATERIALS = {
    "Steel": {
        "strategy":  "prefix",       # match by filename prefix
        "img_dir":   "data/processed/steel_unified/train/images",
        "color":     (200, 100, 30), # BGR blue-ish
        "classes": {
            # display_name : filename_prefix
            "Crazing":          "crazing",
            "Inclusion":        "inclusion",
            "Patches":          "patches",
            "Pitted Surface":   "pitted_surface",
            "Rolled-in Scale":  "rolled-in_scale",
            "Scratches":        "scratches",
        },
    },
    "Aluminum": {
        "strategy":  "label",        # match by reading .txt label files
        "img_dir":   "data/processed_aluminum/images/train",
        "lbl_dir":   "data/processed_aluminum/labels/train",
        "yaml":      "data/dataset_aluminum.yaml",
        "color":     (30, 180, 30),  # BGR green
    },
    "Wood": {
        "strategy":  "label",
        "img_dir":   "data/processed/wood_10class/train/images",
        "lbl_dir":   "data/processed/wood_10class/train/labels",
        "yaml":      "data/dataset_wood_10class.yaml",
        "color":     (30, 90, 160),  # BGR brown
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def sharpness(img_path: Path) -> float:
    """Laplacian variance = higher means sharper."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def best_by_prefix(img_dir: Path, prefix: str, top_n: int = 8) -> Path | None:
    """Pick sharpest image whose stem starts with prefix (no _aug preferred)."""
    p = prefix.lower()
    # prefer originals (no _aug)
    cands = [f for f in img_dir.rglob("*.jpg")
             if f.stem.lower().startswith(p) and "_aug" not in f.stem.lower()]
    if not cands:
        cands = [f for f in img_dir.rglob("*.jpg")
                 if f.stem.lower().startswith(p)]
    if not cands:
        return None
    cands.sort(key=lambda f: f.stat().st_size, reverse=True)
    pool = cands[:top_n]
    pool.sort(key=lambda f: sharpness(f), reverse=True)
    return pool[0]


def build_label_index(img_dir: Path, lbl_dir: Path, class_names: list[str]) -> dict[str, list[Path]]:
    """
    Read every label .txt, find the dominant class_id, group images by class.
    Returns {class_name: [img_path, ...]}
    """
    index: dict[str, list[Path]] = {c: [] for c in class_names}
    for lbl_file in Path(lbl_dir).rglob("*.txt"):
        # find matching image
        for ext in [".jpg", ".jpeg", ".png"]:
            img_path = Path(img_dir) / (lbl_file.stem + ext)
            if img_path.exists():
                break
        else:
            continue
        # read class ids
        try:
            lines = lbl_file.read_text().strip().splitlines()
        except Exception:
            continue
        ids = []
        for line in lines:
            parts = line.split()
            if parts:
                try:
                    ids.append(int(parts[0]))
                except ValueError:
                    pass
        if not ids:
            continue
        # dominant class
        dom = max(set(ids), key=ids.count)
        if 0 <= dom < len(class_names):
            index[class_names[dom]].append(img_path)
    return index


def best_from_pool(pool: list[Path], top_n: int = 8) -> Path | None:
    if not pool:
        return None
    # prefer no _aug, larger file
    orig = [p for p in pool if "_aug" not in p.stem and ".rf." not in p.stem]
    source = orig if orig else pool
    source.sort(key=lambda p: p.stat().st_size, reverse=True)
    candidates = source[:top_n]
    candidates.sort(key=lambda p: sharpness(p), reverse=True)
    return candidates[0]


def make_card(src: Path, display_name: str, material: str, color: tuple) -> np.ndarray:
    """Read image, resize, draw labeled overlay. Returns BGR numpy array."""
    img = cv2.imread(str(src))
    if img is None:
        img = np.zeros((CARD_H, CARD_W, 3), dtype=np.uint8)
    img = cv2.resize(img, (CARD_W, CARD_H), interpolation=cv2.INTER_LANCZOS4)

    # Bottom banner
    banner_h = 72
    overlay = img.copy()
    cv2.rectangle(overlay, (0, CARD_H - banner_h), (CARD_W, CARD_H), (18, 18, 18), -1)
    img = cv2.addWeighted(overlay, 0.80, img, 0.20, 0)

    # Top accent bar (material color)
    cv2.rectangle(img, (0, 0), (CARD_W, 7), color, -1)

    # Class label
    fs, thick = 0.78, 2
    (tw, _), _ = cv2.getTextSize(display_name, FONT, fs, thick)
    tx = (CARD_W - tw) // 2
    cv2.putText(img, display_name, (tx, CARD_H - banner_h + 28),
                FONT, fs, color, thick, cv2.LINE_AA)

    # Material sub-label
    sub = "Material: " + material
    fs2 = 0.45
    (tw2, _), _ = cv2.getTextSize(sub, FONT, fs2, 1)
    cv2.putText(img, sub, ((CARD_W - tw2)//2, CARD_H - banner_h + 56),
                FONT, fs2, (190, 190, 190), 1, cv2.LINE_AA)

    # Team watermark top-right
    wm = "SafePath | RVCE 2026"
    (wmw, _), _ = cv2.getTextSize(wm, FONT, 0.32, 1)
    cv2.putText(img, wm, (CARD_W - wmw - 6, 22), FONT, 0.32, (220, 220, 220), 1, cv2.LINE_AA)

    return img


def build_html(cards: list[dict]) -> str:
    """Print-ready A4 HTML page, 3 cards per row."""
    by_mat: dict[str, list] = {}
    for c in cards:
        by_mat.setdefault(c["material"], []).append(c)

    sections = ""
    mat_colors = {"Steel": "#1e3a5f", "Aluminum": "#1a5c1a", "Wood": "#5c3010"}
    for mat, mat_cards in by_mat.items():
        bg = mat_colors.get(mat, "#333")
        sections += f'<div class="section-header" style="background:{bg}">{mat} Defects</div>\n'
        for i in range(0, len(mat_cards), 3):
            chunk = mat_cards[i:i+3]
            cells = "".join(
                f'<div class="card"><img src="{c["rel"].replace(chr(92), "/")}"><div class="label">{c["label"]}</div><div class="sub">{c["material"]}</div></div>'
                for c in chunk
            )
            sections += f'<div class="row">{cells}</div>\n'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Demo Sample Cards -- Team SafePath RVCE 2026</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Inter',sans-serif;background:#fff;padding:8mm}}
  h1{{text-align:center;font-size:19px;font-weight:700;color:#1a1a2e;margin-bottom:3px}}
  .sub-title{{text-align:center;font-size:11px;color:#555;margin-bottom:10px}}
  .info{{background:#eef2ff;border-left:4px solid #3b5bdb;padding:7px 12px;font-size:10px;color:#333;margin-bottom:12px;border-radius:4px}}
  .section-header{{width:100%;text-align:center;font-size:12px;font-weight:700;color:#fff;border-radius:5px;padding:4px 0;margin:10px 0 5px}}
  .row{{display:flex;gap:7px;margin-bottom:7px;justify-content:center}}
  .card{{width:175px;border:1.5px solid #ddd;border-radius:7px;overflow:hidden;page-break-inside:avoid;box-shadow:0 2px 5px rgba(0,0,0,.08)}}
  .card img{{width:100%;height:155px;object-fit:cover;display:block}}
  .label{{padding:5px 6px 2px;font-size:10px;font-weight:600;color:#1a1a2e;text-align:center}}
  .sub{{padding:0 6px 5px;font-size:8.5px;color:#888;text-align:center}}
  @media print{{body{{padding:4mm}}.card{{box-shadow:none}}h1{{font-size:15px}}}}
</style>
</head>
<body>
  <h1>Defect Sample Cards -- Team SafePath</h1>
  <div class="sub-title">RVCE Hackathon 2026 | Multi-Material Crack Inspection System</div>
  <div class="info">
    <b>How to use:</b> Print on A4 glossy paper. Cut along card borders. Hold each card
    15-30 cm from camera in good lighting. The system will auto-detect material and highlight the defect.<br>
    <b>To print:</b> Open this file in Chrome &gt; Ctrl+P &gt; A4, No margins, Background graphics ON &gt; Print!
  </div>
  {sections}
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  Building Demo Print Cards -- Team SafePath | RVCE 2026")
    print("=" * 62)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_cards: list[dict] = []
    missing:   list[str]  = []

    for material, cfg in MATERIALS.items():
        mat_dir = OUT_DIR / material
        mat_dir.mkdir(exist_ok=True)
        color    = cfg["color"]
        strategy = cfg["strategy"]
        img_dir  = Path(cfg["img_dir"])

        print(f"\n[{material}] strategy={strategy}, dir={img_dir}")

        if not img_dir.exists():
            print(f"  WARNING: {img_dir} not found, skipping.")
            continue

        # ── Prefix strategy (Steel) ──────────────────────────────────────────
        if strategy == "prefix":
            for display_name, prefix in cfg["classes"].items():
                src = best_by_prefix(img_dir, prefix)
                if src is None:
                    print(f"  MISSING: {display_name}")
                    missing.append(f"{material}/{display_name}")
                    continue
                _save_card(src, display_name, material, color, mat_dir, all_cards)
                print(f"  OK  {display_name:22s}  <- {src.name}")

        # ── Label strategy (Aluminum, Wood) ──────────────────────────────────
        elif strategy == "label":
            lbl_dir = Path(cfg["lbl_dir"])
            with open(cfg["yaml"], encoding="utf-8") as f:
                ds = yaml.safe_load(f)
            class_names: list[str] = ds.get("names", [])

            print(f"  Classes ({len(class_names)}): {class_names}")
            print(f"  Indexing labels in {lbl_dir} ...")

            index = build_label_index(img_dir, lbl_dir, class_names)

            for cls_name in class_names:
                pool = index.get(cls_name, [])
                src  = best_from_pool(pool)
                display = cls_name.replace("_", " ").title()
                if src is None:
                    print(f"  MISSING: {display}")
                    missing.append(f"{material}/{display}")
                    continue
                _save_card(src, display, material, color, mat_dir, all_cards)
                print(f"  OK  {display:22s}  <- {src.name}")

    # Write HTML
    html_path = OUT_DIR / "print_sheet.html"
    html_path.write_text(build_html(all_cards), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"  Done! {len(all_cards)} cards built.")
    if missing:
        print(f"  Missing ({len(missing)}): {', '.join(missing)}")
    print(f"\n  Folder : {OUT_DIR.resolve()}")
    print(f"  HTML   : {html_path.resolve()}")
    print("\n  Zip 'demo/print_cards/' and share with your friend.")
    print("  Open print_sheet.html in Chrome -> Ctrl+P -> Print!")
    print("=" * 62)


def _save_card(src, display_name, material, color, mat_dir, all_cards):
    card_img = make_card(src, display_name, material, color)
    safe     = display_name.lower().replace(" ", "_").replace("-", "_")
    out_path = mat_dir / f"{safe}.jpg"
    cv2.imwrite(str(out_path), card_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    rel = str(Path(material) / out_path.name)
    all_cards.append({"rel": rel, "label": display_name, "material": material})


if __name__ == "__main__":
    main()
