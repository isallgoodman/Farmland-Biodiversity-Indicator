# -*- coding: utf-8 -*-
"""
Perennial_3class (>=4/7 consistency) from already-filtered perennial rasters

Inputs: 7 rasters that contain only {90,100,110}.
Keep pixels that have ANY of {90,100,110} in >=4 years.
Output value is the dominant code among {90,100,110}.
All other pixels = NoData.

Output:
D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Perennial_3class.tif
"""

# =========================
# 1) Imports
# =========================
import os
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


# =========================
# 2) Paths
# =========================
INPUT_TIFS = [
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Perennial\Bayern_Perennial_2018.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Perennial\Bayern_Perennial_2019.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Perennial\Bayern_Perennial_2020.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Perennial\Bayern_Perennial_2021.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Perennial\Bayern_Perennial_2022.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Perennial\Bayern_Perennial_2023.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Perennial\Bayern_Perennial_2024.tif",
]

OUT_DIR = r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates"
OUT_TIF = os.path.join(OUT_DIR, "Perennial_3class.tif")


# =========================
# 3) Settings
# =========================
MIN_YEARS = 4
OUT_NODATA = -9999


# =========================
# 4) Helpers
# =========================
def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def check_alignment(dss, paths):
    ref = dss[0]
    for i, ds in enumerate(dss[1:], start=1):
        if ds.count != 1:
            raise ValueError(f"Expected single-band raster: {paths[i]}")
        if (ds.crs != ref.crs or ds.transform != ref.transform or
                ds.width != ref.width or ds.height != ref.height):
            raise ValueError(f"Grid mismatch: {paths[i]}")


def choose_dominant_code(c90, c100, c110):
    """Dominant perennial code; ties -> 90 then 100 then 110."""
    out = np.full(c90.shape, 90, dtype=np.int16)
    best = c90.astype(np.int16)

    m = c100.astype(np.int16) > best
    out[m] = 100
    best[m] = c100.astype(np.int16)[m]

    m = c110.astype(np.int16) > best
    out[m] = 110
    return out


# =========================
# 5) Main (window-wise)
# =========================
def main():
    ensure_dir(OUT_DIR)

    dss = [rasterio.open(p) for p in INPUT_TIFS]
    try:
        check_alignment(dss, INPUT_TIFS)
        ref = dss[0]

        profile = ref.profile.copy()
        profile.update(
            dtype=rasterio.int16,
            count=1,
            nodata=OUT_NODATA,
            compress="LZW",
            predictor=2,
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        total_blocks = sum(1 for _ in ref.block_windows(1))

        with rasterio.open(OUT_TIF, "w", **profile) as dst:
            for _, window in tqdm(ref.block_windows(1), total=total_blocks, desc="Perennial_3class (>=4/7)"):
                stack = np.stack([ds.read(1, window=window).astype(np.int16) for ds in dss], axis=0)

                # count occurrences (NoData is simply "not equal to 90/100/110")
                c90 = np.sum(stack == 90, axis=0, dtype=np.uint8)
                c100 = np.sum(stack == 100, axis=0, dtype=np.uint8)
                c110 = np.sum(stack == 110, axis=0, dtype=np.uint8)

                total = (c90 + c100 + c110).astype(np.uint8)  # 0..7
                keep = total >= MIN_YEARS

                chosen = choose_dominant_code(c90, c100, c110)

                out = np.full(chosen.shape, OUT_NODATA, dtype=np.int16)
                out[keep] = chosen[keep]
                dst.write(out, 1, window=window)

        print(f"\nDone. Output:\n{OUT_TIF}")

    finally:
        for ds in dss:
            ds.close()


if __name__ == "__main__":
    main()