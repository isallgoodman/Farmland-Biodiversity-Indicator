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
OUT_TIF = os.path.join(OUT_DIR, "Functional_Types_9Class.tif")


# ============================================================
# 2) FUNCTIONAL GROUP DEFINITIONS (CROP CODES)
# ============================================================
SPRING_CODES = {
    21,  # Sommerweizen  (spring wheat)
    22,  # Sommergerste  (spring barley)
    23,  # Sommerhafer   (spring oat)
}

LEAF_CODES = {
    30,  # Mais         (maize)
    50,  # Kartoffeln   (potato)
    60,  # Zuckerrüben  (sugar beet)
    71,  # Raps         (rapeseed)
}


# ============================================================
# 3) HELPERS
# ============================================================
def ensure_dir(path_str: str) -> None:
    Path(path_str).mkdir(parents=True, exist_ok=True)


def check_alignment(dss):
    """All rasters must have identical grid: width/height/transform/CRS."""
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
# 4) CORE LOGIC (KEEP NODATA AS NODATA)
# ============================================================
def count_to_bin(count: np.ndarray) -> np.ndarray:
    """
    Convert a 0..7 count (years) into the 3 bins used in the 3x3 matrix:

      bin 0 -> 0/7
      bin 1 -> 1/7 .. 4/7
      bin 2 -> 5/7 .. 7/7
    """
    b = np.zeros_like(count, dtype=np.uint8)
    b[(count >= 1) & (count <= 4)] = 1
    b[count >= 5] = 2
    return b


def functional_types_9class_keep_nodata(stack_3d: np.ndarray, nodata_value) -> np.ndarray:
    """
    stack_3d: (7, H, W) crop codes.

    Valid pixel definition:
      - Pixel is VALID if at least one of the 7 years is not NoData.
      - Pixel is INVALID (background) if all 7 years are NoData -> output NoData.

    For valid pixels:
      - spring_count = number of years in SPRING_CODES (only among valid years)
      - leaf_count   = number of years in LEAF_CODES   (only among valid years)
      - map to 1..9 using the matrix bins:
          class = 1 + spring_bin + 3*leaf_bin
    """
    arr = stack_3d.astype(np.int32, copy=False)

    if nodata_value is None:
        valid_year = np.ones_like(arr, dtype=bool)
    else:
        valid_year = (arr != nodata_value)

    pixel_valid = np.any(valid_year, axis=0)

    is_spring = np.isin(arr, list(SPRING_CODES))
    is_leaf = np.isin(arr, list(LEAF_CODES))

    # counts ONLY where year is valid
    spring_count = np.sum(valid_year & is_spring, axis=0, dtype=np.uint8)
    leaf_count = np.sum(valid_year & is_leaf, axis=0, dtype=np.uint8)

    spring_bin = count_to_bin(spring_count)
    leaf_bin = count_to_bin(leaf_count)

    out = (1 + spring_bin + 3 * leaf_bin).astype(np.int16)

    # keep background as nodata (not 0 / not class 1)
    out[~pixel_valid] = -9999
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

        out_nodata = -9999  # distinct from valid class values 1..9

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
                desc="Computing Functional Types (9 classes, keep NoData)",
            ):
                stack_3d = np.stack([ds.read(1, window=window) for ds in dss], axis=0)
                out = functional_types_9class_keep_nodata(stack_3d, nodata)
                dst.write(out, 1, window=window)

        print(f"\nDone. Output written to:\n{OUT_TIF}")

    finally:
        for ds in dss:
            ds.close()


if __name__ == "__main__":
    main()