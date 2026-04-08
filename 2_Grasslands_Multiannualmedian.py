# -*- coding: utf-8 -*-
"""
Grassland 12-class MULTIANNUAL MEDIAN (2018–2024) - MEMORY SAFE (block-wise)

- Reads MF grid as reference (10 m)
- Reprojects FC onto MF grid per window
- Creates yearly 12-class codes (uint8)
- Computes per-pixel median class across 7 years (ignoring nodata=0)
- Writes ONE output raster

Class logic:
MF bins: 0–2 -> col=1, 3 -> col=2, 4+ -> col=3
FC bins (leap-year aware DOY thresholds):
  <15 May -> row=1
  15–31 May -> row=2
  1–30 Jun -> row=3
  >=1 Jul -> row=4
Class code = row*10 + col  => {11..13,21..23,31..33,41..43}
"""

# =========================
# 1) Imports + Settings
# =========================
import os
from pathlib import Path
from datetime import date

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# =========================
# 2) Input lists (as provided)
# =========================
FC_TIFS = [
    r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\Bayern_FC\Bayern_GRASSLAND_MOW_DE_2018_FIRSTCUT.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\Bayern_FC\Bayern_GRASSLAND_MOW_DE_2019_FIRSTCUT.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\Bayern_FC\Bayern_GRASSLAND_MOW_DE_2020_FIRSTCUT.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\Bayern_FC\Bayern_GRASSLAND_MOW_DE_2021_FIRSTCUT.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\Bayern_FC\Bayern_GRASSLAND_MOW_DE_2022_FIRSTCUT.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\Bayern_FC\Bayern_GRASSLAND_MOW_DE_2023_FIRSTCUT.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\Bayern_FC\Bayern_GRASSLAND_MOW_DE_2024_FIRSTCUT.tif",
]

MF_TIFS = [
    r"D:\The New Folder\Master_arbeit\DLR_Data\MF\Bayern_MF\Bayern_GRASSLAND_MOW_DE_2018_FREQUENCY_10m.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\MF\Bayern_MF\Bayern_GRASSLAND_MOW_DE_2019_FREQUENCY_10m.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\MF\Bayern_MF\Bayern_GRASSLAND_MOW_DE_2020_FREQUENCY_10m.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\MF\Bayern_MF\Bayern_GRASSLAND_MOW_DE_2021_FREQUENCY_10m.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\MF\Bayern_MF\Bayern_GRASSLAND_MOW_DE_2022_FREQUENCY_10m.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\MF\Bayern_MF\Bayern_GRASSLAND_MOW_DE_2023_FREQUENCY_10m.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\MF\Bayern_MF\Bayern_GRASSLAND_MOW_DE_2024_FREQUENCY_10m.tif",
]

# =========================
# 3) Output
# =========================
OUT_DIR = r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Grassland12Classes"
OUT_MEDIAN = os.path.join(OUT_DIR, "Grassland12Class_median_2018_2024.tif")

# =========================
# 4) Helpers
# =========================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def doy(y, m, d):
    return (date(y, m, d) - date(y, 1, 1)).days + 1

def thresholds_for_year(y):
    return {
        "t15may": doy(y, 5, 15),
        "t31may": doy(y, 5, 31),
        "t1jun":  doy(y, 6, 1),
        "t30jun": doy(y, 6, 30),
        "t1jul":  doy(y, 7, 1),
    }

def classify_12_uint8(mf, fc, thr, mf_nodata, fc_nodata):
    """
    mf, fc are 2D arrays on same grid (window).
    Returns uint8 class map with 0 as nodata.
    """
    valid = (mf != mf_nodata) & (fc != fc_nodata)

    out = np.zeros(mf.shape, dtype=np.uint8)

    # MF column
    col = np.zeros(mf.shape, dtype=np.uint8)
    col[mf <= 2] = 1
    col[mf == 3] = 2
    col[mf >= 4] = 3

    # FC row (DOY bins)
    row = np.zeros(fc.shape, dtype=np.uint8)
    row[fc < thr["t15may"]] = 1
    row[(fc >= thr["t15may"]) & (fc <= thr["t31may"])] = 2
    row[(fc >= thr["t1jun"]) & (fc <= thr["t30jun"])] = 3
    row[fc >= thr["t1jul"]] = 4

    ok = valid & (row > 0) & (col > 0)
    out[ok] = (row[ok] * 10 + col[ok]).astype(np.uint8)
    return out

def median_ignore_zeros_uint8(stack_uint8):
    """
    stack_uint8: (N, H, W) uint8 with 0 = nodata
    returns: (H, W) uint8 median ignoring zeros; output 0 if all zeros
    """
    x = stack_uint8.astype(np.uint8)

    # replace 0 with 255 for sorting, then ignore those at the end
    x2 = x.copy()
    x2[x2 == 0] = 255
    x2.sort(axis=0)  # in-place sort along time axis (N=7 small)

    # count valid (non-255) per pixel
    valid_count = np.sum(x2 != 255, axis=0)

    # median index within valid values: (k-1)//2
    k = valid_count
    med_idx = (k - 1) // 2  # integer, works for odd/even by picking lower median

    out = np.zeros(valid_count.shape, dtype=np.uint8)

    # handle pixels with at least 1 valid year
    mask = k > 0
    # gather median values per pixel
    # loop is fine because only 7 years, but do vectorized indexing carefully
    # We'll do it by iterating possible k values (1..7) -> cheap and clean.
    for kk in range(1, 8):
        m = (k == kk)
        if np.any(m):
            idx = (kk - 1) // 2
            out[m] = x2[idx, :, :][m]

    # any remaining would be 0 already
    return out

def make_windows(width, height, block=1024):
    """Generate windows covering raster."""
    for row_off in range(0, height, block):
        h = min(block, height - row_off)
        for col_off in range(0, width, block):
            w = min(block, width - col_off)
            yield Window(col_off, row_off, w, h)

# =========================
# 5) Main
# =========================
def main():
    ensure_dir(OUT_DIR)

    # open all datasets once
    mf_ds_list = [rasterio.open(p) for p in MF_TIFS]
    fc_ds_list = [rasterio.open(p) for p in FC_TIFS]

    try:
        # reference grid = MF 2018 (assumes all MF aligned; if not, you should fix MF first)
        ref = mf_ds_list[0]
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

        # nodata handling
        mf_nodata = ref.nodata if ref.nodata is not None else 0
        # for FC, we reproject to a temp array using a chosen nodata:
        fc_nodata = 0  # safe because FC DOY is 1..365/366

        # output profile
        out_profile = ref.profile.copy()
        out_profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            nodata=0,
            compress="LZW",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        with rasterio.open(OUT_MEDIAN, "w", **out_profile) as out_ds:
            windows = list(make_windows(ref_width, ref_height, block=1024))

            for win in tqdm(windows, desc="Computing median 12-class (block-wise)"):
                # per-window yearly class maps (7, h, w)
                cls_stack = np.zeros((7, int(win.height), int(win.width)), dtype=np.uint8)

                for i, y in enumerate(YEARS):
                    mf_ds = mf_ds_list[i]
                    fc_ds = fc_ds_list[i]

                    # read MF window
                    mf = mf_ds.read(1, window=win)

                    # reproject FC into MF window grid
                    fc_dst = np.full((int(win.height), int(win.width)), fc_nodata, dtype=np.float32)

                    # destination transform for this window
                    win_transform = rasterio.windows.transform(win, ref_transform)

                    reproject(
                        source=rasterio.band(fc_ds, 1),
                        destination=fc_dst,
                        src_transform=fc_ds.transform,
                        src_crs=fc_ds.crs,
                        src_nodata=fc_ds.nodata,
                        dst_transform=win_transform,
                        dst_crs=ref_crs,
                        dst_nodata=fc_nodata,
                        resampling=Resampling.nearest,
                    )

                    # convert FC to integer DOY (nearest already, but stored float)
                    fc = fc_dst.astype(np.int32)

                    thr = thresholds_for_year(y)
                    cls = classify_12_uint8(mf, fc, thr, mf_nodata=mf_nodata, fc_nodata=fc_nodata)
                    cls_stack[i] = cls

                # median across years ignoring zeros
                med = median_ignore_zeros_uint8(cls_stack)

                # write block
                out_ds.write(med, 1, window=win)

        print(f"\nDone. Saved:\n{OUT_MEDIAN}")

    finally:
        for ds in mf_ds_list:
            ds.close()
        for ds in fc_ds_list:
            ds.close()

if __name__ == "__main__":
    main()