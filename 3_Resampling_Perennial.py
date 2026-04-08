from pathlib import Path
import numpy as np
import rasterio

IN_TIF  = r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Perennial_3class.tif"
OUT_DIR = r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled"
OUT_TIF = str(Path(OUT_DIR) / "Perennial_3class_weighted_float.tif")

# mapping: class_value -> score
MAP = {
    90: 0.25,    # vineyard
    110: 0.125,  # hops
    100: 0.875,  # fruits & woody vegetation
}

FALLBACK_NODATA = -9999.0

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    with rasterio.open(IN_TIF) as src:
        arr = src.read(1)
        nodata = src.nodata
        out_nodata = float(nodata) if nodata is not None else FALLBACK_NODATA

        out = np.full(arr.shape, out_nodata, dtype=np.float32)

        # keep nodata as nodata, recode only valid pixels
        valid = np.ones(arr.shape, dtype=bool)
        if nodata is not None:
            valid &= (arr != nodata)

        for k, v in MAP.items():
            out[valid & (arr == k)] = np.float32(v)

        # (optional) warn if there are unexpected class values (excluding nodata)
        unexpected = np.unique(arr[valid & ~np.isin(arr, list(MAP.keys()))])
        if unexpected.size > 0:
            print(f"Warning: unexpected class values found (left as NoData): {unexpected[:50]}")

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