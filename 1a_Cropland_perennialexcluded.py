# -*- coding: utf-8 -*-
"""
Perennial persistence mask (2018–2024):

Goal:
- Look at 7 yearly crop-type rasters (already Bayern, EPSG:32632, 10m).
- Perennial classes are: 90, 100, 110.
- Identify pixels that are perennial in >4 years (i.e., at least 4 of 7 years; 50%+).
- EXCLUDE those pixels from ALL yearly rasters by setting them to NoData.
- Write cleaned rasters to a new folder.

Notes:
- Assumes all 7 rasters are aligned (same shape/transform/CRS). This should be true
  because you produced them with the same clip+reproject+10m pipeline.
"""

# =========================
# 1) Imports + Settings
# =========================
import os
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

PERENNIAL_CODES = (90, 100, 110)
MIN_YEARS = 4  # "more than 4 years (50%)" interpreted as >=4 out of 7

# =========================
# 2) Paths (EDIT IF NEEDED)
# =========================
IN_DIR = r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_Cropland"
OUT_DIR = r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_Cropland_perennialExcluded"

# If your filenames differ, adjust the pattern.
# Expected from prior step: Bayern_croptypes_YYYY_10m_cropland.tif
PATTERN = "Bayern_croptypes_*_10m_cropland.tif"

# =========================
# 3) Helpers
# =========================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def is_aligned(meta_a, meta_b) -> bool:
    keys = ["crs", "transform", "width", "height"]
    return all(meta_a[k] == meta_b[k] for k in keys)

# =========================
# 4) Main
# =========================
def main():
    ensure_dir(OUT_DIR)

    files = sorted(Path(IN_DIR).glob(PATTERN))
    if len(files) != 7:
        raise ValueError(f"Expected 7 rasters, found {len(files)} in {IN_DIR} matching {PATTERN}")

    # ---- 4.1 Read all rasters and build perennial count
    count = None
    ref_profile = None
    nodata = None

    for i, f in enumerate(tqdm(files, desc="Building perennial frequency mask")):
        with rasterio.open(f) as ds:
            a = ds.read(1)

            if i == 0:
                ref_profile = ds.profile.copy()
                nodata = ds.nodata if ds.nodata is not None else 0
                count = np.zeros(a.shape, dtype=np.uint8)
            else:
                if not is_aligned(ref_profile, ds.profile):
                    raise ValueError(f"Raster grid mismatch: {f.name} is not aligned with the first raster.")

            perennial = np.isin(a, PERENNIAL_CODES)
            count += perennial.astype(np.uint8)

    # Pixels to exclude: perennial in >= MIN_YEARS out of 7
    exclude_mask = count >= MIN_YEARS  # True where "persistent perennial"

    # ---- 4.2 Apply exclusion to every raster and write outputs
    for f in tqdm(files, desc="Excluding persistent perennial pixels from all years"):
        out_path = str(Path(OUT_DIR) / f.name.replace("_cropland.tif", "_cropland_perennialExcluded.tif"))

        with rasterio.open(f) as ds:
            a = ds.read(1)

            # Set those pixels to NoData (exclude them from all years)
            a_out = a.copy()
            a_out[exclude_mask] = nodata

            out_profile = ds.profile.copy()
            out_profile.update(
                compress="LZW",
                tiled=True,
                blockxsize=256,
                blockysize=256,
                nodata=nodata,
            )

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(a_out, 1)

    # ---- 4.3 Save the mask itself (optional but useful for reproducibility)
    mask_path = str(Path(OUT_DIR) / "perennial_persistent_mask_ge4of7.tif")
    mask_profile = ref_profile.copy()
    mask_profile.update(
        dtype="uint8",
        count=1,
        nodata=0,
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(exclude_mask.astype(np.uint8), 1)

    print("\nDone.")
    print(f"Inputs:  {IN_DIR}")
    print(f"Outputs: {OUT_DIR}")
    print(f"Saved mask: {mask_path}")

if __name__ == "__main__":
    main()