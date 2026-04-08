import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


# =========================
# Inputs / Outputs
# =========================
INPUT_TIFS = [
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_Croptype\Bayern_Croptypes_2018.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_Croptype\Bayern_Croptypes_2019.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_Croptype\Bayern_Croptypes_2020.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_Croptype\Bayern_Croptypes_2021.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_Croptype\Bayern_Croptypes_2022.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_Croptype\Bayern_Croptypes_2023.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_Croptype\Bayern_Croptypes_2024.tif",
]

OUT_DIR = r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates"
OUT_TIF = os.path.join(OUT_DIR, "Crop_Count.tif")


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def check_alignment(dss):
    """Ensure all rasters match grid exactly."""
    ref = dss[0]
    for i, ds in enumerate(dss[1:], start=1):
        if (ds.width != ref.width or ds.height != ref.height):
            raise ValueError(f"Raster size mismatch: {INPUT_TIFS[i]}")
        if (ds.transform != ref.transform):
            raise ValueError(f"Raster transform mismatch: {INPUT_TIFS[i]}")
        if (ds.crs != ref.crs):
            raise ValueError(f"Raster CRS mismatch: {INPUT_TIFS[i]}")
        if ds.count != 1:
            raise ValueError(f"Expected single-band raster: {INPUT_TIFS[i]}")


def unique_count_window(stack_3d: np.ndarray, nodata_value) -> np.ndarray:
    """
    stack_3d: shape (7, H, W)
    Returns uint8 array (H, W) with unique counts in [0..7].
    0 means all years nodata.
    """
    # Choose a sentinel that sorts before valid crop codes (crop codes are typically non-negative)
    sentinel = np.int32(-32768)

    arr = stack_3d.astype(np.int32, copy=False)

    if nodata_value is not None:
        invalid = (arr == nodata_value)
    else:
        invalid = np.zeros_like(arr, dtype=bool)

    arr2 = arr.copy()
    arr2[invalid] = sentinel

    # Sort values along time axis
    s = np.sort(arr2, axis=0)  # (7, H, W)
    valid = (s != sentinel)

    # Count unique values among valid entries:
    # A new unique starts at i if:
    # - it's valid AND (previous is invalid OR value differs from previous)
    change0 = valid[0]
    change_rest = valid[1:] & (~valid[:-1] | (s[1:] != s[:-1]))
    count = change0.astype(np.uint8) + np.sum(change_rest, axis=0, dtype=np.uint8)

    # Pixels with all invalid become 0 automatically
    return count.astype(np.uint8)


def main():
    ensure_dir(OUT_DIR)

    dss = [rasterio.open(p) for p in INPUT_TIFS]
    try:
        check_alignment(dss)

        ref = dss[0]
        nodata = ref.nodata

        profile = ref.profile.copy()
        profile.update(
            dtype=rasterio.uint8,
            count=1,
            nodata=0,           # 0 = no valid years
            compress="LZW",
            predictor=2,
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        total_blocks = sum(1 for _ in ref.block_windows(1))

        with rasterio.open(OUT_TIF, "w", **profile) as dst:
            for _, window in tqdm(ref.block_windows(1), total=total_blocks, desc="Computing Crop_Count"):
                stack = []
                for ds in dss:
                    a = ds.read(1, window=window)
                    stack.append(a)
                stack_3d = np.stack(stack, axis=0)  # (7, h, w)

                out = unique_count_window(stack_3d, nodata)
                dst.write(out, 1, window=window)

        print(f"\nDone. Output written to:\n{OUT_TIF}")

    finally:
        for ds in dss:
            ds.close()


if __name__ == "__main__":
    main()