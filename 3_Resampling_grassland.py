import os
import numpy as np
import rasterio
from tqdm import tqdm

in_path = r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Grassland_12class_median.tif"
out_path = r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled\Grasslands.tif"

os.makedirs(os.path.dirname(out_path), exist_ok=True)

# 41 -> 1.0, 31 -> 0.875, 21 -> 0.75, 11 -> 0.625, 42/32 -> 0.5, rest -> 0.375
def remap(arr, nodata_value):
    out = np.full(arr.shape, 0.375, dtype=np.float32)
    out[arr == 41] = 1.0
    out[arr == 31] = 0.875
    out[arr == 21] = 0.75
    out[arr == 11] = 0.625
    out[(arr == 42) | (arr == 32)] = 0.5
    if nodata_value is not None:
        out[arr == nodata_value] = np.float32(nodata_value)
    return out

with rasterio.open(in_path) as src:
    profile = src.profile.copy()

    in_nodata = src.nodata
    out_nodata = np.float32(in_nodata) if in_nodata is not None else np.nan

    profile.update(
        driver="GTiff",
        dtype=rasterio.float32,
        nodata=out_nodata,
        count=1,
        compress=profile.get("compress", "deflate"),
        predictor=profile.get("predictor", 3),
        tiled=profile.get("tiled", True),
        BIGTIFF="IF_SAFER",
    )

    windows = list(src.block_windows(1))

    with rasterio.open(out_path, "w", **profile) as dst:
        for _, window in tqdm(windows, total=len(windows), desc="Reclassifying Grasslands"):
            data = src.read(1, window=window, masked=True)

            filled = data.filled(in_nodata if in_nodata is not None else -9999999)

            out = remap(filled, in_nodata)

            if np.ma.isMaskedArray(data) and np.any(data.mask):
                out = np.ma.array(out, mask=data.mask).filled(out_nodata)

            dst.write(out.astype(np.float32), 1, window=window)

print("Wrote:", out_path)