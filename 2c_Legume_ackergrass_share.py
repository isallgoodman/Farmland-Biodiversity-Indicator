import os
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


# ============================================================
# 1) INPUTS / OUTPUTS
# ============================================================
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
OUT_TIF = os.path.join(OUT_DIR, "Legume_Ackergrass_Share.tif")


# ============================================================
# 2) TARGET CODES (LEGUME / TEMP GRASSLAND)
# ============================================================
TARGET_CODES = {
    40,  # Bohnen/Lupinen/Erbsen  (legumes)
    81,  # Klee/Luzerne          (clover/alfalfa)
    82,  # Ackergras             (arable grass / temporary grassland)
}


# ============================================================
# 3) HELPERS
# ============================================================
def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def check_alignment(dss):
    """Ensure all rasters match grid exactly."""
    ref = dss[0]
    for i, ds in enumerate(dss[1:], start=1):
        if (ds.width != ref.width) or (ds.height != ref.height):
            raise ValueError(f"Raster size mismatch: {INPUT_TIFS[i]}")
        if ds.transform != ref.transform:
            raise ValueError(f"Raster transform mismatch: {INPUT_TIFS[i]}")
        if ds.crs != ref.crs:
            raise ValueError(f"Raster CRS mismatch: {INPUT_TIFS[i]}")
        if ds.count != 1:
            raise ValueError(f"Expected single-band raster: {INPUT_TIFS[i]}")


# ============================================================
# 4) CORE LOGIC
# ============================================================
def compute_share_count(stack_3d: np.ndarray, nodata_value) -> np.ndarray:
    """
    stack_3d: shape (7, H, W) crop codes.

    Definitions:
      - VALID pixel: at least one year is not NoData
      - Output value: count of years in {40,81,82} => 0..7 for valid pixels
      - Pixels invalid in all years (NoData all 7 years): output NoData

    Returns:
      out: int16 (H, W), with NoData where invalid-all-years, else 0..7
    """
    arr = stack_3d.astype(np.int32, copy=False)

    if nodata_value is None:
        # If the rasters have no nodata metadata, treat everything as valid
        valid_year = np.ones_like(arr, dtype=bool)
    else:
        valid_year = (arr != nodata_value)

    # Pixel valid if any year is valid
    pixel_valid = np.any(valid_year, axis=0)

    # Target membership (regardless of nodata; we mask with valid_year)
    is_target = np.isin(arr, list(TARGET_CODES))

    # Count years where (valid AND target)
    cnt = np.sum(valid_year & is_target, axis=0, dtype=np.int16)  # 0..7

    # Set invalid-all-years pixels to nodata in output
    out = cnt.copy()
    out[~pixel_valid] = -9999  # temporary nodata (will be set in profile)
    return out


# ============================================================
# 5) MAIN (BLOCK-WINDOW PROCESSING)
# ============================================================
def main():
    ensure_dir(OUT_DIR)

    dss = [rasterio.open(p) for p in INPUT_TIFS]
    try:
        check_alignment(dss)

        ref = dss[0]
        nodata = ref.nodata

        out_nodata = -9999  # keep separate from "0 years" which is a valid value

        profile = ref.profile.copy()
        profile.update(
            dtype=rasterio.int16,
            count=1,
            nodata=out_nodata,
            compress="LZW",
            predictor=2,
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        total_blocks = sum(1 for _ in ref.block_windows(1))

        with rasterio.open(OUT_TIF, "w", **profile) as dst:
            for _, window in tqdm(
                ref.block_windows(1),
                total=total_blocks,
                desc="Computing Legume/Ackergrass share (0..7 on valid pixels)",
            ):
                stack_3d = np.stack([ds.read(1, window=window) for ds in dss], axis=0)

                out = compute_share_count(stack_3d, nodata)
                dst.write(out, 1, window=window)

        print(f"\nDone. Output written to:\n{OUT_TIF}")

    finally:
        for ds in dss:
            ds.close()


if __name__ == "__main__":
    main()