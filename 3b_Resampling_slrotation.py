# 3e_recode_functional_types_9class_to_score.py
# Recode Functional_Types_9Class.tif classes to float scores and write to Intermediates_resampled.
#
# Rules (user-defined):
#   1 -> 0.000
#   3 -> 0.125
#   7 -> 0.125
#   9 -> 0.250
#   2 -> 0.375
#   4 -> 0.375
#   6 -> 0.500
#   8 -> 0.500
#   5 -> 0.625
#
# Output:
#   float32 raster, NoData preserved (or fallback), same grid/metadata as input.

from pathlib import Path
import numpy as np
import rasterio


# ============================================================
# 1) INPUTS / OUTPUTS
# ============================================================
IN_TIF = r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Functional_Types_9Class.tif"
OUT_DIR = r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled"
OUT_NAME = "Functional_Types_9Class_score_float.tif"
OUT_TIF = str(Path(OUT_DIR) / OUT_NAME)

# If input has no nodata defined, use this for output
FALLBACK_NODATA = -9999.0


# ============================================================
# 2) RECODE TABLE (EDIT HERE)
# ============================================================
RECODE_MAP = {
    1: 0.0,
    3: 0.125,
    7: 0.125,
    9: 0.125,
    2: 0.25,
    4: 0.25,
    6: 0.25,
    8: 0.25,
    5: 0.50,
}


# ============================================================
# 3) HELPERS
# ============================================================
def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def make_valid_mask(arr: np.ndarray, nodata) -> np.ndarray:
    """
    Valid pixels = not nodata (and not NaN for float inputs).
    """
    valid = np.ones(arr.shape, dtype=bool)
    if nodata is not None:
        valid &= (arr != nodata)
    if np.issubdtype(arr.dtype, np.floating):
        valid &= ~np.isnan(arr)
    return valid


def warn_unexpected_values(arr: np.ndarray, valid: np.ndarray, allowed: set) -> None:
    """
    Prints a warning if there are unexpected class values (excluding nodata).
    """
    uniq = np.unique(arr[valid])
    unexpected = uniq[~np.isin(uniq, list(allowed))]
    if unexpected.size > 0:
        preview = unexpected[:50]
        print(f"Warning: unexpected class values found (left as NoData): {preview}")


# ============================================================
# 4) MAIN
# ============================================================
def main() -> None:
    ensure_dir(OUT_DIR)

    with rasterio.open(IN_TIF) as src:
        arr = src.read(1)
        nodata = src.nodata
        out_nodata = float(nodata) if nodata is not None else float(FALLBACK_NODATA)

        valid = make_valid_mask(arr, nodata)

        # Initialize output as NoData everywhere
        out = np.full(arr.shape, out_nodata, dtype=np.float32)

        # Apply recode map
        for k, v in RECODE_MAP.items():
            out[valid & (arr == k)] = np.float32(v)

        # Optional diagnostics
        warn_unexpected_values(arr, valid, allowed=set(RECODE_MAP.keys()))

        # Write output
        meta = src.meta.copy()
        meta.update(
            dtype="float32",
            count=1,
            nodata=out_nodata,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

        with rasterio.open(OUT_TIF, "w", **meta) as dst:
            dst.write(out, 1)

    print(f"Done.\nSaved: {OUT_TIF}")


if __name__ == "__main__":
    main()