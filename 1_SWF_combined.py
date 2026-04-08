# -*- coding: utf-8 -*-
"""
SWF (binary: 1 or NoData):
1) Reproject to EPSG:32632 at 10 m
2) Clip to Bayern
3) Union (pixel-wise MAX) -> one combined SWF

Outputs are binary:
"""

# =========================
# 1) Imports
# =========================
import os
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping
from tqdm import tqdm

# =========================
# 2) Paths + Settings
# =========================
INPUT_TIFS = [
    r"D:\The New Folder\Master_arbeit\DLR_Data\SWF\SWF_2015_5m.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\SWF\SWF_2018_5m.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\SWF\SWF_2021_5m.tif",
]

BAYERN_SHP = r"D:\The New Folder\MASTErARBEIT\Shapefiles\Bayern.shp"
OUT_DIR = r"D:\The New Folder\Master_arbeit\DLR_Data\SWF\Bayern_clipped"

TARGET_CRS = "EPSG:32632"
TARGET_RES = 10  # meters
NODATA = 0       # stored as nodata; valid data is 1 only (binary)

COMBINED_OUT = os.path.join(OUT_DIR, "Bayern_SWF_Combined_2015_2018_2021_10m.tif")

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# =========================
# 3) Bayern geometry (EPSG:32632)
# =========================
gdf = gpd.read_file(BAYERN_SHP).to_crs(TARGET_CRS)
bayern_geom = gdf.geometry.union_all()
bayern_shapes = [mapping(bayern_geom)]

# =========================
# 4) Reproject (10 m) + Clip each raster
# =========================
clipped_files = []
ref_meta = None

for tif in tqdm(INPUT_TIFS, desc="Reproject(10m) + clip SWF to Bayern (EPSG:32632)"):
    in_name = Path(tif).name
    out_path = os.path.join(OUT_DIR, "Bayern_" + in_name.replace("_5m", "_10m"))

    with rasterio.open(tif) as src:
        # ---- 4.1 compute 10 m target grid in EPSG:32632
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            TARGET_CRS,
            src.width,
            src.height,
            *src.bounds,
            resolution=TARGET_RES
        )

        # ---- 4.2 reproject into an array (nearest, binary-safe)
        dst_arr = np.full((1, dst_height, dst_width), NODATA, dtype=np.uint8)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_arr[0],
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata if src.nodata is not None else NODATA,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            dst_nodata=NODATA,
            resampling=Resampling.nearest,
        )

        # ---- 4.3 wrap as an in-memory dataset so mask() can clip it
        meta = {
            "driver": "GTiff",
            "height": dst_height,
            "width": dst_width,
            "count": 1,
            "dtype": "uint8",
            "crs": TARGET_CRS,
            "transform": dst_transform,
            "nodata": NODATA,
        }

        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**meta) as tmp:
                tmp.write(dst_arr)

                clipped, clipped_transform = mask(
                    tmp,
                    bayern_shapes,
                    crop=True,
                    nodata=NODATA,
                    filled=True
                )

                out_meta = meta.copy()
                out_meta.update({
                    "height": clipped.shape[1],
                    "width": clipped.shape[2],
                    "transform": clipped_transform,
                    "compress": "LZW",
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                })

    # Enforce strict binary output: values are {1, NODATA}
    clipped = np.where(clipped == 1, 1, NODATA).astype(np.uint8)

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(clipped)

    if ref_meta is None:
        ref_meta = out_meta.copy()

    clipped_files.append(out_path)

# =========================
# 5) Union (pixel-wise MAX) -> Combined SWF (binary 1 or NoData)
# =========================
combined = None

for f in tqdm(clipped_files, desc="Union (MAX) across years"):
    with rasterio.open(f) as ds:
        a = ds.read(1).astype(np.uint8)
        a = np.where(a == 1, 1, 0).astype(np.uint8)  # treat nodata as 0 for combining
        combined = a if combined is None else np.maximum(combined, a)

# Convert back to {1, NODATA}
combined = np.where(combined == 1, 1, NODATA).astype(np.uint8)

# Write combined with same grid/metadata as first clipped output
ref_meta.update({"count": 1, "dtype": "uint8", "nodata": NODATA})

with rasterio.open(COMBINED_OUT, "w", **ref_meta) as dst:
    dst.write(combined, 1)

print("\nDone.")
print("Clipped (10 m) files:")
for f in clipped_files:
    print(" -", f)
print("Combined union (10 m):")
print(" -", COMBINED_OUT)
