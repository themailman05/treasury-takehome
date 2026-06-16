"""
Generate a test set of ~150 alcohol-beverage labels with Gemini ("nano banana"),
plus a ground-truth manifest for the label-verification prototype.

Why this isn't just a 150-iteration loop
-----------------------------------------
A test set for a *verification* tool needs more than pretty labels:

  1. Ground truth. Every image is paired in `manifest.jsonl` / `manifest.csv` with
     the "application" field values an agent would have on screen, so the verifier
     has something to compare against.
  2. A deliberate mix of compliant and DEFECTIVE labels, with defect categories that
     map 1:1 to what the verifier checks (missing warning clause, title-case header,
     reworded warning, absent warning, ABV mismatch, brand variant, net-contents
     mismatch). Without defects you can only test `pass`, never `fail`/`needs_review`.
  3. Photo-condition variety (clean / angled / glare / low-light / blur) to exercise
     Jenny's imperfect-image requirement and the OCR robustness path.

Each manifest row records the application values, what the label actually renders
(which differs for mismatch defects), and the EXPECTED verdict — so it doubles as an
assertion table for the verifier's test harness.

Caveat on text fidelity
------------------------
Image models render text well now but not perfectly, especially a ~40-word legal
paragraph. Treat generated labels as approximate and QA them. For pixel-true warning
ground truth (to unit-test verify_warning precisely), prefer the hybrid approach:
generate the background art here, then composite exact text with Pillow. Ask if you
want that variant — this script is the "fast, varied, realistic" path the brief
suggests.

Usage
-----
    pip install google-genai python-dotenv pillow
    export GEMINI_API_KEY=...                 # or put it in a .env file
    python generate_labels.py                 # full run (~150 images)
    python generate_labels.py --count 20      # smaller batch
    python generate_labels.py --dry-run       # specs + manifest only, no API calls
    python generate_labels.py --model gemini-3-pro-image-preview   # max text fidelity

Cost/throughput: ~$0.04-0.15 per image depending on model/resolution, so a 150-image
run is roughly $6-23. All Gemini images carry an invisible SynthID watermark.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from io import BytesIO
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

OUTPUT_DIR = Path("samples")
IMAGE_DIR = OUTPUT_DIR / "images"
MANIFEST_JSONL = OUTPUT_DIR / "manifest.jsonl"
MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"

DEFAULT_MODEL = "gemini-3.1-flash-image"   # "nano banana 2": fast + strong text
DEFAULT_COUNT = 150
SEED = 1788                                # reproducible spec generation
MAX_WORKERS = 4
MAX_RETRIES = 4

# Canonical government warning (27 CFR 16.21) — keep in sync with verify_warning.py
WARNING_CANONICAL = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
    "drink alcoholic beverages during pregnancy because of the risk of birth "
    "defects. (2) Consumption of alcoholic beverages impairs your ability to drive "
    "a car or operate machinery, and may cause health problems."
)
WARNING_CLAUSE_1 = WARNING_CANONICAL.split("(2)")[0].strip()  # header + clause 1
WARNING_TITLE_CASE = WARNING_CANONICAL.replace("GOVERNMENT WARNING:", "Government Warning:")
WARNING_REWORDED = WARNING_CANONICAL.replace(
    "should not drink alcoholic beverages during pregnancy",
    "are advised to avoid alcohol while pregnant",
)


# --------------------------------------------------------------------------- #
# Beverage catalog
# --------------------------------------------------------------------------- #

# (beverage_type, class/type designation, (abv_min, abv_max), shows_proof)
BEVERAGES = [
    ("bourbon",      "Kentucky Straight Bourbon Whiskey", (40, 50), True),
    ("scotch",       "Single Malt Scotch Whisky",         (40, 46), True),
    ("gin",          "London Dry Gin",                    (40, 47), True),
    ("vodka",        "Premium Vodka",                     (40, 40), True),
    ("rum",          "Aged Caribbean Rum",                (40, 45), True),
    ("tequila",      "Tequila Reposado",                  (38, 40), True),
    ("red_wine",     "Cabernet Sauvignon",                (13, 15), False),
    ("white_wine",   "Chardonnay",                        (12, 14), False),
    ("ipa",          "India Pale Ale",                    (5, 8),   False),
    ("lager",        "Pilsner Lager",                     (4, 6),   False),
]

NET_CONTENTS = ["750 mL", "375 mL", "1 L", "500 mL", "355 mL", "50 mL"]

BRAND_FIRST = ["Old Tom", "Iron Crest", "Silver Hollow", "Copper", "Stone's",
               "Black Fox", "Gold Ridge", "Crooked", "Cedar", "Highland",
               "Blue Heron", "Red Barn", "Wandering", "Granite", "Wild Oak"]
BRAND_SECOND = ["Distillery", "Reserve", "& Oak", "Throw", "Cellars", "Works",
                "Brewing Co.", "Vineyards", "Spirits", "Trading Co.", "House",
                "Barrel Co.", "Estate", "Provisions"]

STYLES = ["vintage letterpress", "minimalist modern", "ornate Victorian",
          "rustic kraft-paper", "art-deco gold-foil", "clean Scandinavian",
          "hand-drawn botanical", "bold industrial"]

PHOTO_CONDITIONS = {
    "clean_studio":  "sharp studio product photograph, even lighting, straight-on, white background",
    "angled":        "photographed at a 25-degree angle on a wooden table, natural light",
    "glare":         "photographed under harsh light with visible glare and a bright highlight on the glass",
    "low_light":     "photographed in dim warm ambient light, slightly underexposed",
    "slight_blur":   "handheld phone snapshot, slightly soft focus, mild motion blur",
}

# Scenario -> (warning_state, field_override_kind, expected_overall, note)
# warning_state in {compliant, missing_clause, title_case, reworded, absent}
SCENARIOS = [
    ("compliant",            "compliant",      None,            "pass",         "fully compliant"),
    ("warning_missing_clause","missing_clause", None,           "fail",         "second warning clause dropped"),
    ("warning_title_case",   "title_case",      None,           "fail",         "GOVERNMENT WARNING not in caps"),
    ("warning_reworded",     "reworded",        None,           "fail",         "warning wording altered"),
    ("warning_absent",       "absent",          None,           "fail",         "no government warning present"),
    ("abv_mismatch",         "compliant",       "abv",          "fail",         "label ABV differs from application"),
    ("net_contents_mismatch","compliant",       "net_contents", "fail",         "label net contents differ from application"),
    ("brand_variant",        "compliant",       "brand_case",   "needs_review", "brand casing/punctuation variant"),
    ("class_type_mismatch",  "compliant",       "class_type",   "fail",         "label class/type differs from application"),
]

# Weighted scenario sampling: majority compliant, the rest spread across defects.
SCENARIO_WEIGHTS = {
    "compliant": 0.40,
    "warning_missing_clause": 0.09,
    "warning_title_case": 0.09,
    "warning_reworded": 0.07,
    "warning_absent": 0.05,
    "abv_mismatch": 0.09,
    "net_contents_mismatch": 0.07,
    "brand_variant": 0.08,
    "class_type_mismatch": 0.06,
}


# --------------------------------------------------------------------------- #
# Spec generation
# --------------------------------------------------------------------------- #

@dataclass
class LabelSpec:
    id: str
    beverage_type: str
    scenario: str
    photo_condition: str
    style: str
    # application values (what the agent's COLA application says)
    brand_name: str
    class_type: str
    abv: str               # numeric string, e.g. "45"
    abv_display: str        # e.g. "45% Alc./Vol. (90 Proof)"
    net_contents: str
    # what the label actually renders (differs from application for mismatch defects)
    label_brand: str
    label_class_type: str
    label_abv_display: str
    label_net_contents: str
    warning_state: str
    warning_text: str
    # expected outcome for the verifier test harness
    expected_overall: str
    expected_note: str
    prompt: str = ""

    def filename(self) -> str:
        return f"{self.id}_{self.beverage_type}_{self.scenario}.png"


def _abv_display(abv: int, shows_proof: bool) -> str:
    if shows_proof:
        return f"{abv}% Alc./Vol. ({abv * 2} Proof)"
    return f"{abv}% Alc./Vol."


def _warning_for_state(state: str) -> str:
    return {
        "compliant": WARNING_CANONICAL,
        "missing_clause": WARNING_CLAUSE_1,
        "title_case": WARNING_TITLE_CASE,
        "reworded": WARNING_REWORDED,
        "absent": "",
    }[state]


def _sample_scenarios(rng: random.Random, count: int) -> list[str]:
    """Deterministic count per scenario, summing exactly to `count`."""
    names = list(SCENARIO_WEIGHTS)
    counts = {n: int(SCENARIO_WEIGHTS[n] * count) for n in names}
    while sum(counts.values()) < count:          # distribute rounding remainder
        counts[rng.choice(names)] += 1
    bag = [n for n, c in counts.items() for _ in range(c)]
    rng.shuffle(bag)
    return bag


def build_specs(count: int, seed: int = SEED) -> list[LabelSpec]:
    rng = random.Random(seed)
    scenario_bag = _sample_scenarios(rng, count)
    cond_names = list(PHOTO_CONDITIONS)
    specs: list[LabelSpec] = []

    for i, scenario in enumerate(scenario_bag):
        bev, class_type, (lo, hi), proof = rng.choice(BEVERAGES)
        abv = rng.randint(lo, hi)
        abv_disp = _abv_display(abv, proof)
        net = rng.choice(NET_CONTENTS)
        brand = f"{rng.choice(BRAND_FIRST)} {rng.choice(BRAND_SECOND)}"
        style = rng.choice(STYLES)
        # compliant labels mostly clean; defects spread across conditions
        cond = "clean_studio" if (scenario == "compliant" and rng.random() < 0.5) \
            else rng.choice(cond_names)

        _, warning_state, override, expected_overall, note = next(
            s for s in SCENARIOS if s[0] == scenario
        )

        label_brand, label_class, label_abv, label_net = brand, class_type, abv_disp, net
        if override == "abv":
            wrong = abv + rng.choice([-2, -1, 1, 2])
            label_abv = _abv_display(wrong, proof)
        elif override == "net_contents":
            label_net = rng.choice([n for n in NET_CONTENTS if n != net])
        elif override == "brand_case":
            # Dave's "STONE'S THROW" vs "Stone's Throw" — same brand, different render
            label_brand = brand.upper()
        elif override == "class_type":
            # label declares a different class/type than the application states
            label_class = rng.choice([c for _, c, _, _ in BEVERAGES if c != class_type])

        spec = LabelSpec(
            id=f"L{i:03d}",
            beverage_type=bev,
            scenario=scenario,
            photo_condition=cond,
            style=style,
            brand_name=brand,
            class_type=class_type,
            abv=str(abv),
            abv_display=abv_disp,
            net_contents=net,
            label_brand=label_brand,
            label_class_type=label_class,
            label_abv_display=label_abv,
            label_net_contents=label_net,
            warning_state=warning_state,
            warning_text=_warning_for_state(warning_state),
            expected_overall=expected_overall,
            expected_note=note,
        )
        spec.prompt = build_prompt(spec)
        specs.append(spec)
    return specs


def build_prompt(s: LabelSpec) -> str:
    condition = PHOTO_CONDITIONS[s.photo_condition]
    if s.warning_state == "absent":
        warning_instr = "Do NOT include any government warning text anywhere on the label."
    else:
        warning_instr = (
            "On the back/lower portion of the label, in small but legible print, "
            f'include this exact government warning text verbatim:\n"{s.warning_text}"'
        )
    return (
        f"A realistic, high-resolution {condition} of a {s.beverage_type.replace('_', ' ')} "
        f"bottle whose front label uses a {s.style} design. The label clearly shows, "
        f"as crisp legible text:\n"
        f'- Brand name: "{s.label_brand}"\n'
        f'- Product type: "{s.label_class_type}"\n'
        f'- Alcohol content: "{s.label_abv_display}"\n'
        f'- Net contents: "{s.label_net_contents}"\n'
        f"{warning_instr}\n"
        f"Spell all text exactly as written. The label should look like a genuine "
        f"commercial product, not a mockup or template. No extra captions or borders."
    )


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def write_manifest(specs: list[LabelSpec]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with MANIFEST_JSONL.open("w") as f:
        for s in specs:
            row = asdict(s)
            row["filename"] = s.filename()
            f.write(json.dumps(row) + "\n")
    fields = ["id", "filename", "beverage_type", "scenario", "expected_overall",
              "expected_note", "brand_name", "class_type", "abv", "abv_display",
              "net_contents", "label_brand", "label_class_type", "label_abv_display",
              "label_net_contents", "warning_state", "photo_condition"]
    with MANIFEST_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for s in specs:
            row = asdict(s)
            row["filename"] = s.filename()
            w.writerow(row)
    print(f"Wrote manifest: {MANIFEST_JSONL} and {MANIFEST_CSV} ({len(specs)} rows)")


def _save_image_from_response(response, dest: Path) -> bool:
    """Pull the first inline image part out of a generate_content response."""
    from PIL import Image
    parts = getattr(response, "parts", None) or response.candidates[0].content.parts
    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline is not None and getattr(inline, "data", None):
            Image.open(BytesIO(inline.data)).save(dest)
            return True
    return False


def generate_one(client, model: str, spec: LabelSpec) -> tuple[str, str]:
    dest = IMAGE_DIR / spec.filename()
    if dest.exists():                       # resumable: skip already-generated
        return spec.id, "skipped"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(model=model, contents=[spec.prompt])
            if _save_image_from_response(resp, dest):
                return spec.id, "ok"
            return spec.id, "no_image_in_response"
        except Exception as e:              # noqa: BLE001 - includes rate limits
            if attempt == MAX_RETRIES:
                return spec.id, f"error: {type(e).__name__}: {e}"
            time.sleep(2 ** attempt)        # exponential backoff
    return spec.id, "error: exhausted retries"


def run(model: str, count: int, dry_run: bool) -> None:
    specs = build_specs(count)
    write_manifest(specs)

    # Quick distribution summary
    from collections import Counter
    dist = Counter(s.scenario for s in specs)
    print("Scenario distribution:")
    for name, n in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {name:24s} {n}")

    if dry_run:
        print("\n--dry-run: skipping API calls. Sample prompt:\n")
        print(specs[0].prompt)
        return

    try:
        from google import genai
    except ImportError:
        sys.exit("Install the SDK first: pip install google-genai python-dotenv pillow")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment or .env")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    client = genai.Client()
    print(f"\nGenerating {len(specs)} labels with {model} ({MAX_WORKERS} workers)...")

    results = Counter()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(generate_one, client, model, s): s for s in specs}
        for done, fut in enumerate(as_completed(futures), 1):
            spec_id, status = fut.result()
            results[status.split(":")[0]] += 1
            tag = status if status in ("ok", "skipped") else f"FAIL ({status})"
            print(f"[{done:3d}/{len(specs)}] {spec_id} -> {tag}")

    print("\nDone:", dict(results))
    print(f"Images in {IMAGE_DIR}/, ground truth in {MANIFEST_JSONL}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate alcohol-label test set with Gemini")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="gemini-3.1-flash-image (nano banana 2), gemini-2.5-flash-image "
                        "(original), or gemini-3-pro-image-preview (Pro, best text)")
    p.add_argument("--count", type=int, default=DEFAULT_COUNT)
    p.add_argument("--dry-run", action="store_true",
                   help="Build specs + manifest, print a sample prompt, make no API calls")
    args = p.parse_args()
    run(args.model, args.count, args.dry_run)


if __name__ == "__main__":
    main()
