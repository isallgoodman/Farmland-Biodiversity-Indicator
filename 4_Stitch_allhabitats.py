# Stitch.py
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm

paths = [
    r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled\Grasslands.tif",       # master + first priority
    r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled\Bayern_SWF.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled\Cropland_median.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled\Perennial.tif",
]

out_dir = r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled"
os.makedirs(out_dir, exist_ok=True)
out_mosaic = os.path.join(out_dir, "Habitat_stitched.tif")

OUT_NODATA = -9999.0


def main():
    with rasterio.open(paths[0]) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        profile = ref.profile.copy()

    profile.update(
        driver="GTiff",
        dtype=rasterio.float32,
        count=1,
        crs=ref_crs,
        transform=ref_transform,
        width=ref_width,
        height=ref_height,
        nodata=OUT_NODATA,
        compress=profile.get("compress", "deflate"),
        predictor=profile.get("predictor", 3),
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )

    dsets = [rasterio.open(p) for p in paths]
    try:
        with rasterio.open(out_mosaic, "w", **profile) as dst:
            windows = list(dst.block_windows(1))

            for _, window in tqdm(windows, total=len(windows), desc="Stitching 4 habitat rasters"):
                h, w = window.height, window.width
                win_transform = rasterio.windows.transform(window, ref_transform)

                out = np.full((h, w), OUT_NODATA, dtype=np.float32)
                filled = np.zeros((h, w), dtype=bool)

                for ds in dsets:
                    tmp = np.full((h, w), OUT_NODATA, dtype=np.float32)

                    reproject(
                        source=rasterio.band(ds, 1),
                        destination=tmp,
                        src_transform=ds.transform,
                        src_crs=ds.crs,
                        src_nodata=ds.nodata,
                        dst_transform=win_transform,
                        dst_crs=ref_crs,
                        dst_nodata=OUT_NODATA,
                        resampling=Resampling.nearest,
                    )

                    valid = ~np.isclose(tmp, OUT_NODATA)
                    take = valid & (~filled)
                    if np.any(take):
                        out[take] = tmp[take]
                        filled[take] = True

                    if filled.all():
                        break

                dst.write(out, 1, window=window)

    finally:
        for ds in dsets:
            ds.close()

    print("Wrote:", out_mosaic)


if __name__ == "__main__":
    main()