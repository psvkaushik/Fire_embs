"""
Locates SMARTIES' cached trust_remote_code source files and greps them for
the actual sensor-band -> generic-spectral-group mapping (e.g. what makes
Sentinel-2's blue B02 map to "blue_1" vs "blue_2" vs "blue_3"), so we can
hardcode the CORRECT CaBuAr S2-id -> SMARTIES-name mapping instead of
guessing from wavelength order alone.

Usage:
    python find_smarties_mapping.py
"""

import glob
import os
import re

# trust_remote_code=True caches the model's custom .py files here
cache_globs = [
    os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/**/*.py"),
    os.path.expanduser("~/.cache/huggingface/hub/**/*.py"),
]

found = []
for pattern in cache_globs:
    found += glob.glob(pattern, recursive=True)

# keep only files that look SMARTIES-related
found = [f for f in found if "smarties" in f.lower() or "SMARTIES" in open(f, errors="ignore").read()[:2000].lower()]

print(f"Found {len(found)} candidate file(s):")
for f in found:
    print(f"  {f}")
print()

KEYWORDS = ["sentinel", "s2", "blue_", "green_", "red_", "near_infrared", "short_wave_infrared",
            "projection_conversion", "spectrum_spec", "aerosol", "band_name", "BAND"]

for f in found:
    try:
        text = open(f, errors="ignore").read()
    except Exception as e:
        continue
    hits = [kw for kw in KEYWORDS if kw.lower() in text.lower()]
    if not hits:
        continue
    print(f"=== {f} ===  (matched: {hits})")
    for i, line in enumerate(text.splitlines()):
        if any(kw.lower() in line.lower() for kw in KEYWORDS):
            print(f"  {i+1:5d}: {line.strip()}")
    print()