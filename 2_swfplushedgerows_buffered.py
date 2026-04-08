import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


# ============================================================
# 1) INPUTS / OUTPUTS
# ============================================================
HEDGEROWS_GPKG = r"D:\The New Folder\Master_arbeit\DLR_Data\Hedgegrows\HedgeRows_Bavaria.gpkg"
SWF_TIF = r"D:\The New Folder\Master_arbeit\DLR_Data\SWF\Bayern_clipped\Bayern_SWF_Combined_2015_2018_2021_10m.tif"
CROPTYPE_TIF = r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_croptypes_2023.tif"

OUT_DIR = r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates"
OUT_TIF = os.path.join(OUT_DIR, "Bayern_SWF.tif")

# Distance threshold (meters)
MAX_DIST_M = 100.0

# Output nodata (float)
OUT_NODATA = -9999.0

# Rasterize behavior
ALL_TOUCHED = False  # set True if you want thicker hedgerow coverage


# ============================================================
# 2) HELPERS
# ============================================================
def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def to_bool_swf(arr: np.ndarray, nodata) -> np.ndarray:
    """
    Convert SWF raster to boolean presence.
    Assumption:
      - SWF presence is indicated by values > 0.
      - nodata (if defined) is invalid.
    """
    if nodata is None:
        return arr > 0
    return (arr != nodata) & (arr > 0)


def load_hedgerows_as_target_crs(gpkg_path: str, target_crs) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg_path)
    if gdf.crs is None:
        raise ValueError("HedgeRows_Bavaria.gpkg has no CRS. Define it before running.")
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf


def rasterize_hedgerows_to_match(gdf: gpd.GeoDataFrame, ref_ds: rasterio.io.DatasetReader) -> np.ndarray:
    shapes = (
        (geom, 1)
        for geom in gdf.geometry
        if geom is not None and (not geom.is_empty)
    )
    out = rasterize(
        shapes=shapes,
        out_shape=(ref_ds.height, ref_ds.width),
        transform=ref_ds.transform,
        fill=0,
        dtype="uint8",
        all_touched=ALL_TOUCHED,
    )
    return out


def is_farmland(crop_arr: np.ndarray, crop_nodata) -> np.ndarray:
    """
    User rule:
      - ALL pixels in CropTypes are farmland
      - so farmland = everything that is NOT NoData
    """
    if crop_nodata is None:
        # If no nodata is defined, treat all pixels as farmland
        return np.ones(crop_arr.shape, dtype=bool)
    return crop_arr != crop_nodata


def compute_within_distance_mask(
    crop_path: str,
    target_crs,
    target_transform,
    target_shape,
    max_dist_m: float
) -> np.ndarray:
    """
    Returns boolean mask on the TARGET grid where pixels are within max_dist_m of farmland.
    CropTypes is reprojected onto the target grid first (nearest).
    """
    tgt_h, tgt_w = target_shape

    with rasterio.open(crop_path) as crop_ds:
        crop = crop_ds.read(1)
        crop_nodata = crop_ds.nodata

        # Reproject crop onto target grid
        fill_val = crop_nodata if crop_nodata is not None else 0
        crop_on_tgt = np.full((tgt_h, tgt_w), fill_val, dtype=crop.dtype)

        reproject(
            source=crop,
            destination=crop_on_tgt,
            src_transform=crop_ds.transform,
            src_crs=crop_ds.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
            src_nodata=crop_nodata,
            dst_nodata=fill_val,
        )

        farmland = is_farmland(crop_on_tgt, crop_nodata)

    # Pixel size (meters) from target transform (assumes projected CRS)
    px_w = abs(target_transform.a)
    px_h = abs(target_transform.e)
    px_m = (px_w + px_h) / 2.0

    # Distance to nearest farmland:
    # distance_transform_edt computes distance to nearest ZERO,
    # so pass (~farmland) where farmland=True => 0 targets.
    dist_pix = distance_transform_edt(~farmland)
    dist_m = dist_pix * px_m

    return dist_m <= max_dist_m


# ============================================================
# 3) STEP A: READ SWF + BUILD SWF PRESENCE MASK
# ============================================================
def build_swf_mask(swf_ds: rasterio.io.DatasetReader) -> tuple[np.ndarray, np.ndarray]:
    swf = swf_ds.read(1)
    swf_nodata = swf_ds.nodata

    swf_bool = to_bool_swf(swf, swf_nodata)

    if swf_nodata is not None:
        swf_invalid = (swf == swf_nodata)
    else:
        swf_invalid = np.zeros(swf.shape, dtype=bool)

    return swf_bool, swf_invalid


# ============================================================
# 4) STEP B: RASTERIZE HEDGEROWS TO SWF GRID
# ============================================================
def build_hedgerow_mask(gpkg_path: str, swf_ds: rasterio.io.DatasetReader) -> np.ndarray:
    gdf = load_hedgerows_as_target_crs(gpkg_path, swf_ds.crs)
    hedgerow_r = rasterize_hedgerows_to_match(gdf, swf_ds)  # 0/1
    return hedgerow_r == 1


# ============================================================
# 5) STEP C: FARMLAND VICINITY MASK (<= 100 m)
# ============================================================
def build_farmland_vicinity_mask(swf_ds: rasterio.io.DatasetReader) -> np.ndarray:
    return compute_within_distance_mask(
        crop_path=CROPTYPE_TIF,
        target_crs=swf_ds.crs,
        target_transform=swf_ds.transform,
        target_shape=(swf_ds.height, swf_ds.width),
        max_dist_m=MAX_DIST_M,
    )


# ============================================================
# 6) STEP D: COMBINE + APPLY VICINITY + WRITE FLOAT OUTPUT
# ============================================================
def combine_apply_write(
    swf_ds: rasterio.io.DatasetReader,
    swf_bool: np.ndarray,
    swf_invalid: np.ndarray,
    hedgerow_bool: np.ndarray,
    near_farmland: np.ndarray,
    out_tif: str,
) -> None:
    combined = (swf_bool | hedgerow_bool) & near_farmland

    out = np.full((swf_ds.height, swf_ds.width), OUT_NODATA, dtype=np.float32)
    out[combined] = 1.0
    out[swf_invalid] = OUT_NODATA  # preserve SWF nodata as nodata

    profile = swf_ds.profile.copy()
    profile.update(
        dtype=rasterio.float32,
        count=1,
        nodata=float(OUT_NODATA),
        compress="LZW",
        predictor=2,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(out, 1)


# ============================================================
# 7) MAIN
# ============================================================
def main():
    ensure_dir(OUT_DIR)

    with rasterio.open(SWF_TIF) as swf_ds:
        # Step A
        swf_bool, swf_invalid = build_swf_mask(swf_ds)

        # Step B
        hedgerow_bool = build_hedgerow_mask(HEDGEROWS_GPKG, swf_ds)

        # Step C
        near_farmland = build_farmland_vicinity_mask(swf_ds)

        # Step D
        combine_apply_write(
            swf_ds=swf_ds,
            swf_bool=swf_bool,
            swf_invalid=swf_invalid,
            hedgerow_bool=hedgerow_bool,
            near_farmland=near_farmland,
            out_tif=OUT_TIF,
        )

    print(f"\nDone. Output written to:\n{OUT_TIF}")


if __name__ == "__main__":
    main()