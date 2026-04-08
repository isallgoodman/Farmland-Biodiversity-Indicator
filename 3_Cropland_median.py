# Cropland_median.py
# Compute pixelwise MEDIAN across 3 input rasters (per-pixel: take the median of the 3 values).
# - Uses available (non-nodata / non-masked) values only.
# - If all 3 are nodata for a pixel -> output nodata.
# - No snapping, no max.
#
# Output:
#   D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled\Cropland_median.tif

import os
import numpy as np
import rasterio
from tqdm import tqdm


# ============================================================
# 1) INPUTS / OUTPUTS
# ============================================================
paths = [
    r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled\Crop_Count.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled\Functional_Classes.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled\Legume_Ackergrass.tif",
]

out_dir = r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled"
os.makedirs(out_dir, exist_ok=True)

out_med = os.path.join(out_dir, "Cropland_median.tif")

OUT_NODATA = -9999.0


# ============================================================
# 2) GRID CHECKS
# ============================================================
def assert_same_grid(dsets):
    ref = dsets[0]
    for i, ds in enumerate(dsets[1:], start=1):
        if ds.crs != ref.crs:
            raise ValueError(f"CRS mismatch: dataset 0 vs dataset {i}")
        if ds.transform != ref.transform:
            raise ValueError(f"Transform mismatch: dataset 0 vs dataset {i}")
        if ds.width != ref.width or ds.height != ref.height:
            raise ValueError(f"Shape mismatch: dataset 0 vs dataset {i}")


# ============================================================
# 3) READ HELPERS
# ============================================================
def read_as_nan(ds, window):
    """Read band 1 as float32 and convert nodata/mask to NaN."""
    arr = ds.read(1, window=window, masked=True)
    a = np.asarray(arr, dtype=np.float32)

    # mask -> NaN
    if np.ma.isMaskedArray(arr) and np.any(arr.mask):
        a[np.asarray(arr.mask)] = np.nan

    # explicit nodata -> NaN
    nd = ds.nodata
    if nd is not None:
        nd_f = np.float32(nd)
        if np.isnan(nd_f):
            a[np.isnan(a)] = np.nan
        else:
            a[np.isclose(a, nd_f, equal_nan=False)] = np.nan

    return a


# ============================================================
# 4) MAIN
# ============================================================
def main():
    dsets = [rasterio.open(p) for p in paths]
    try:
        assert_same_grid(dsets)
        ref = dsets[0]

        profile = ref.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=OUT_NODATA,
            compress=profile.get("compress", "deflate"),
            predictor=profile.get("predictor", 3),
            tiled=profile.get("tiled", True),
        )

        windows = list(ref.block_windows(1))

        with rasterio.open(out_med, "w", **profile) as dst_med:
            for _, window in tqdm(windows, total=len(windows), desc="Cropland median (3 rasters)"):
                stack = np.stack([read_as_nan(ds, window) for ds in dsets], axis=0)  # (3,h,w)

                has_any = np.isfinite(stack).any(axis=0)

                out = np.full(has_any.shape, OUT_NODATA, dtype=np.float32)

                if np.any(has_any):
                    vals = stack[:, has_any]  # (3, N_valid)
                    out[has_any] = np.nanmedian(vals, axis=0).astype(np.float32)

                dst_med.write(out, 1, window=window)

        print("Wrote:", out_med)

    finally:
        for ds in dsets:
            ds.close()


if __name__ == "__main__":
    main()