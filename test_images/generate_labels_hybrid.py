"""
Hybrid label generator: Gemini ("nano banana") paints the artwork, Pillow composites
the exact text and the photo degradations.

Why hybrid
----------
The pure-generation script (generate_labels.py) is fast and realistic but the model
renders text approximately, so a "compliant" label might contain a subtly garbled
warning. That's fine for exercising OCR tolerance, but useless as ground truth for
unit-testing verify_warning — you can't assert a pass/fail if you don't know exactly
what pixels say.

This script splits the concerns:
  - Gemini generates a TEXT-FREE decorative label background (a small reusable pool,
    so you pay for ~12 images, not 150).
  - Pillow draws every character of the brand, class/type, ABV, net contents, and the
    government warning — so the manifest text is, by construction, exactly what's on
    the image. The warning header is rendered bold + caps per 27 CFR 16.21; defects
    (title case, dropped clause, reworded, absent) are produced precisely.
  - Pillow also applies the photo conditions deterministically (perspective tilt,
    glare, low light, blur) instead of asking the model — fully reproducible, and it
    exercises the same OCR-robustness path.

Result: a controllable test set whose manifest is a trustworthy assertion table.

Shares spec/defect/manifest logic with generate_labels.py so the two stay in sync.

Usage
-----
    pip install google-genai python-dotenv pillow numpy
    export GEMINI_API_KEY=...
    python generate_labels_hybrid.py                 # gen bg pool + composite 150
    python generate_labels_hybrid.py --count 20
    python generate_labels_hybrid.py --bg-pool 8     # number of unique backgrounds
    python generate_labels_hybrid.py --procedural-bg # no API calls; PIL paper texture
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# Reuse the ground-truth/defect engine so both generators agree.
from generate_labels import build_specs, LabelSpec, DEFAULT_MODEL  # noqa: E402

OUTPUT_DIR = Path("samples_hybrid")
IMAGE_DIR = OUTPUT_DIR / "images"
BG_DIR = OUTPUT_DIR / "backgrounds"
MANIFEST_JSONL = OUTPUT_DIR / "manifest.jsonl"
MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"

CANVAS = (1024, 1536)        # portrait label
MARGIN = 120                  # neutral border around the label (room for tilt)
SEED = 1788

BG_STYLES = ["vintage letterpress cream paper", "minimalist matte off-white",
             "ornate victorian damask", "rustic kraft paper", "art-deco gold and black",
             "clean scandinavian pale", "botanical engraving border", "weathered parchment"]


# --------------------------------------------------------------------------- #
# Fonts
# --------------------------------------------------------------------------- #

def _find_font(*needles: str) -> str | None:
    roots = ["/usr/share/fonts", "/usr/local/share/fonts",
             os.path.expanduser("~/.fonts")]
    try:
        import matplotlib
        roots.append(os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data/fonts/ttf"))
    except Exception:
        pass
    cands = []
    for r in roots:
        cands += glob.glob(os.path.join(r, "**", "*.ttf"), recursive=True)
    for needle_set in (needles, ("DejaVuSans",), ("LiberationSans",)):
        matches = [c for c in cands
                   if all(n.lower() in os.path.basename(c).lower() for n in needle_set)]
        if matches:
            # Prefer the closest name: a bare needle like "DejaVuSans" should pick
            # DejaVuSans.ttf, not DejaVuSans-BoldOblique.ttf. Shortest basename wins.
            return min(matches, key=lambda c: len(os.path.basename(c)))
    return None

REGULAR_FONT = _find_font("DejaVuSans") or _find_font("LiberationSans-Regular")
BOLD_FONT = _find_font("DejaVuSans-Bold") or _find_font("LiberationSans-Bold") or REGULAR_FONT


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = BOLD_FONT if bold else REGULAR_FONT
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


# --------------------------------------------------------------------------- #
# Background pool (Gemini or procedural)
# --------------------------------------------------------------------------- #

def _procedural_bg(seed: int) -> Image.Image:
    rng = random.Random(seed)
    base = rng.choice([(247, 242, 230), (250, 250, 248), (236, 226, 206),
                       (245, 240, 245), (230, 232, 235)])
    img = Image.new("RGB", CANVAS, base)
    d = ImageDraw.Draw(img)
    # subtle speckle + border to read as "paper", not a flat fill
    for _ in range(4000):
        x, y = rng.randrange(CANVAS[0]), rng.randrange(CANVAS[1])
        j = rng.randint(-10, 10)
        d.point((x, y), fill=tuple(max(0, min(255, c + j)) for c in base))
    bw = rng.randint(8, 16)
    inset = 40
    d.rectangle([inset, inset, CANVAS[0] - inset, CANVAS[1] - inset],
                outline=(120, 105, 80), width=bw)
    return img.filter(ImageFilter.GaussianBlur(0.4))


def generate_bg_pool(model: str, n: int, procedural: bool) -> list[Image.Image]:
    BG_DIR.mkdir(parents=True, exist_ok=True)
    pool: list[Image.Image] = []
    if procedural:
        for i in range(n):
            img = _procedural_bg(SEED + i)
            img.save(BG_DIR / f"bg_{i:02d}.png")
            pool.append(img)
        print(f"Built {n} procedural backgrounds")
        return pool

    from google import genai
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY), or use --procedural-bg")
    client = genai.Client()
    for i in range(n):
        dest = BG_DIR / f"bg_{i:02d}.png"
        if dest.exists():
            pool.append(Image.open(dest).convert("RGB").resize(CANVAS))
            continue
        style = BG_STYLES[i % len(BG_STYLES)]
        prompt = (
            f"A flat 2D rectangular paper label texture in a {style} style, scanned "
            f"top-down so it fills the entire frame edge to edge. Ornamental border and "
            f"paper texture only. ABSOLUTELY NO TEXT, no words, no letters, no numbers. "
            f"NOT a photograph of a bottle, NOT a 3D object, NO bottle, NO glass, NO "
            f"shelf, NO background scene, NO depth of field — just the flat label artwork "
            f"itself. Leave the center large, open and uncluttered for text to be added "
            f"later."
        )
        for attempt in range(1, 5):
            try:
                resp = client.models.generate_content(model=model, contents=[prompt])
                parts = getattr(resp, "parts", None) or resp.candidates[0].content.parts
                saved = False
                for part in parts:
                    inline = getattr(part, "inline_data", None)
                    if inline is not None and getattr(inline, "data", None):
                        img = Image.open(BytesIO(inline.data)).convert("RGB").resize(CANVAS)
                        img.save(dest)
                        pool.append(img)
                        saved = True
                        break
                if saved:
                    print(f"bg {i+1}/{n} ok")
                    break
            except Exception as e:  # noqa: BLE001
                if attempt == 4:
                    print(f"bg {i+1}/{n} failed ({e}); using procedural fallback")
                    img = _procedural_bg(SEED + i)
                    img.save(dest)
                    pool.append(img)
                else:
                    time.sleep(2 ** attempt)
    return pool


# --------------------------------------------------------------------------- #
# Text compositing
# --------------------------------------------------------------------------- #

def _wrap(draw: ImageDraw.ImageDraw, text: str, fnt, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=fnt) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _avg_luma(img: Image.Image, box) -> float:
    import numpy as np
    return float(np.asarray(img.crop(box).convert("L")).mean())


def _centered(draw, text, fnt, y, fill, max_w):
    for line in _wrap(draw, text, fnt, max_w):
        w = draw.textlength(line, font=fnt)
        draw.text(((CANVAS[0] - w) / 2, y), line, font=fnt, fill=fill)
        y += int(fnt.size * 1.25)
    return y


def _block_height(draw, items, max_w: int) -> int:
    """Total pixel height of a stack of (text, font, gap_after) entries."""
    h = 0
    for text, fnt, gap in items:
        h += len(_wrap(draw, text, fnt, max_w)) * int(fnt.size * 1.25) + gap
    return h


def compose_label(spec: LabelSpec, bg: Image.Image) -> Image.Image:
    img = bg.convert("RGB").resize(CANVAS).copy()
    draw = ImageDraw.Draw(img)
    inner = int(CANVAS[0] * 0.70)

    # The Gemini backgrounds carry an ornamental frame whose scrollwork intrudes on
    # the upper-center, so the brand block is drawn inside its own light contrast
    # panel (like the warning) — legible on any background, no collision with art.
    items = [
        (spec.label_brand,        font(72, bold=True), 24),
        (spec.label_class_type,   font(38),            36),
        (spec.label_abv_display,  font(38),            6),
        (spec.label_net_contents, font(32),            0),
    ]
    pad = 34
    block_h = _block_height(draw, items, inner)
    top = int(CANVAS[1] * 0.16)
    panel_w = inner + 2 * pad
    panel_x = (CANVAS[0] - panel_w) // 2

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rounded_rectangle(
        [panel_x, top - pad, panel_x + panel_w, top + block_h + pad],
        radius=16, fill=(252, 250, 245, 224))
    img.paste(overlay, (0, 0), overlay)
    draw = ImageDraw.Draw(img)

    fill = "#1c1a17"
    y = top
    for text, fnt, gap in items:
        y = _centered(draw, text, fnt, y, fill, inner) + gap

    if spec.warning_state != "absent":
        _draw_warning(img, draw, spec)
    return img


def _draw_warning(img, draw, spec: LabelSpec):
    """Header (bold, per the rendered casing) + body, inside a light contrast box
    so it's legible and 'separate and apart' as 27 CFR 16.21 requires."""
    header = "Government Warning:" if spec.warning_state == "title_case" else "GOVERNMENT WARNING:"
    body = spec.warning_text.split(":", 1)[1].strip()  # clauses after the header

    pad, box_w = 28, int(CANVAS[0] * 0.84)
    box_x = (CANVAS[0] - box_w) // 2
    hf, bf = font(26, bold=True), font(24)
    body_lines = _wrap(draw, body, bf, box_w - 2 * pad)
    box_h = pad * 2 + int(hf.size * 1.3) + int(len(body_lines) * bf.size * 1.25)
    box_y = CANVAS[1] - box_h - 70

    # contrast panel
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle([box_x, box_y, box_x + box_w, box_y + box_h],
                         radius=14, fill=(252, 250, 245, 235))
    img.paste(overlay, (0, 0), overlay)
    draw = ImageDraw.Draw(img)

    ty = box_y + pad
    draw.text((box_x + pad, ty), header, font=hf, fill="#141414")
    ty += int(hf.size * 1.35)
    for line in body_lines:
        draw.text((box_x + pad, ty), line, font=bf, fill="#1a1a1a")
        ty += int(bf.size * 1.25)


# --------------------------------------------------------------------------- #
# Photo-condition degradations (deterministic)
# --------------------------------------------------------------------------- #

def _find_coeffs(src, dst):
    import numpy as np
    m = []
    for (sx, sy), (dx, dy) in zip(src, dst):
        m.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        m.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
    A = np.array(m, dtype=float)
    B = np.array(src, dtype=float).reshape(8)
    return np.linalg.solve(A, B).tolist()


def apply_condition(label: Image.Image, condition: str, seed: int) -> Image.Image:
    rng = random.Random(seed)
    canvas = Image.new("RGB", (CANVAS[0] + 2 * MARGIN, CANVAS[1] + 2 * MARGIN), (244, 244, 242))
    canvas.paste(label, (MARGIN, MARGIN))
    W, H = canvas.size

    if condition == "angled":
        try:
            dx, dy = rng.randint(40, 90), rng.randint(20, 50)
            dst = [(dx, dy), (W - dx // 2, 0), (W, H), (0, H - dy)]
            src = [(0, 0), (W, 0), (W, H), (0, H)]
            canvas = canvas.transform((W, H), Image.PERSPECTIVE,
                                      _find_coeffs(src, dst), Image.BICUBIC,
                                      fillcolor=(244, 244, 242))
        except Exception:
            canvas = canvas.rotate(rng.uniform(-7, 7), expand=False, fillcolor=(244, 244, 242))
    elif condition == "glare":
        ov = Image.new("L", (W, H), 0)
        gd = ImageDraw.Draw(ov)
        cx, cy = rng.randint(W // 4, 3 * W // 4), rng.randint(H // 5, H // 2)
        r = rng.randint(W // 5, W // 3)
        gd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=200)
        ov = ov.filter(ImageFilter.GaussianBlur(r // 2))
        white = Image.new("RGB", (W, H), (255, 255, 255))
        canvas = Image.composite(white, canvas, ov)
    elif condition == "low_light":
        canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.45, 0.6))
        canvas = ImageEnhance.Contrast(canvas).enhance(0.9)
    elif condition == "slight_blur":
        canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(1.2, 2.2)))
    # clean_studio: leave as-is
    return canvas


# --------------------------------------------------------------------------- #
# Manifest + driver
# --------------------------------------------------------------------------- #

def write_manifest(specs: list[LabelSpec]) -> None:
    rows = []
    for s in specs:
        r = asdict(s)
        r.pop("prompt", None)               # not used by the hybrid renderer
        r["filename"] = s.filename()
        rows.append(r)
    with MANIFEST_JSONL.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    fields = ["id", "filename", "beverage_type", "scenario", "expected_overall",
              "expected_note", "brand_name", "class_type", "abv", "abv_display",
              "net_contents", "label_brand", "label_class_type", "label_abv_display",
              "label_net_contents", "warning_state", "photo_condition"]
    with MANIFEST_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote manifest ({len(rows)} rows): {MANIFEST_JSONL}")


def run(model: str, count: int, bg_pool_n: int, procedural: bool) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    specs = build_specs(count, seed=SEED)
    write_manifest(specs)

    pool = generate_bg_pool(model, bg_pool_n, procedural)
    if not pool:
        sys.exit("No backgrounds available")

    rng = random.Random(SEED)
    results = Counter()
    for i, spec in enumerate(specs, 1):
        dest = IMAGE_DIR / spec.filename()
        if dest.exists():
            results["skipped"] += 1
            continue
        bg = pool[rng.randrange(len(pool))]
        label = compose_label(spec, bg)
        final = apply_condition(label, spec.photo_condition, SEED + i)
        final.save(dest)
        results["ok"] += 1
        if i % 25 == 0 or i == len(specs):
            print(f"[{i}/{len(specs)}] composited")
    print("Done:", dict(results))
    print(f"Images in {IMAGE_DIR}/, ground truth in {MANIFEST_JSONL}")


def main() -> None:
    p = argparse.ArgumentParser(description="Hybrid alcohol-label generator (Gemini art + Pillow text)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--count", type=int, default=150)
    p.add_argument("--bg-pool", type=int, default=12, help="number of unique backgrounds to reuse")
    p.add_argument("--procedural-bg", action="store_true", help="skip Gemini; use PIL paper texture")
    args = p.parse_args()
    run(args.model, args.count, args.bg_pool, args.procedural_bg)


if __name__ == "__main__":
    main()
