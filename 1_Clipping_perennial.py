# -*- coding: utf-8 -*-
"""
Perennial habitat from CropTypes (Bayern_croptypes_YYYY.tif):

- Keep ONLY class codes: 90, 100, 110
- Everything else -> NoData
- EPSG:32632
- Resample to 10 m x 10 m (nearest neighbour)
- Clip to Bayern extent
"""

# =========================
# 1) Imports + Settings
# =========================
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform
from tqdm import tqdm

TARGET_EPSG = 32632
TARGET_RES = 10  # meters
KEEP_CODES = {90, 100, 110}

# =========================
# 2) Paths (EDIT IF NEEDED)
# =========================
INPUT_TIFS = [
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_croptypes_2018.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_croptypes_2019.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_croptypes_2020.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_croptypes_2021.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_croptypes_2022.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_croptypes_2023.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\Bayern_croptypes_2024.tif",
]

BAYERN_SHP = r"D:\The New Folder\MASTErARBEIT\Shapefiles\Bayern.shp"

OUT_DIR = r"D:\The New Folder\Master_arbeit\DLR_Data\Crop_Type\perennial"

# =========================
# 3) Helpers
# =========================
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def load_bayern_geometry_32632(shp_path: str):
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"No features found in: {shp_path}")
    gdf = gdf.to_crs(epsg=TARGET_EPSG)
    geom = gdf.geometry.union_all()
    return [geom]

def process_one(in_tif: str, out_tif: str, bayern_geom_32632):
    with rasterio.open(in_tif) as src:
        nodata = src.nodata if src.nodata is not None else 0

        # Define a 10m EPSG:32632 grid covering the source bounds
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            f"EPSG:{TARGET_EPSG}",
            src.width,
            src.height,
            *src.bounds,
            resolution=(TARGET_RES, TARGET_RES),
        )

        # Reproject + resample on-the-fly (nearest for categorical codes)
        with WarpedVRT(
            src,
            crs=f"EPSG:{TARGET_EPSG}",
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            resampling=Resampling.nearest,
            nodata=nodata,
        ) as vrt:

            out_image, out_transform = mask(
                vrt,
                bayern_geom_32632,
                crop=True,
                all_touched=False,
                filled=True,
                nodata=nodata,
            )

            # Keep ONLY 90/100/110, everything else -> nodata
            band = out_image[0]
            keep_mask = np.isin(band, list(KEEP_CODES))
            band = np.where(keep_mask, band, nodata).astype(band.dtype)
            out_image[0] = band

            out_meta = vrt.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "crs": f"EPSG:{TARGET_EPSG}",
                    "nodata": nodata,
                    "compress": "LZW",
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                }
            )

            with rasterio.open(out_tif, "w", **out_meta) as dst:
                dst.write(out_image)

# =========================
# 4) Main
# =========================
def main():
    ensure_dir(OUT_DIR)
    bayern_geom_32632 = load_bayern_geometry_32632(BAYERN_SHP)

    for in_tif in tqdm(INPUT_TIFS, desc="Perennial (90/100/110) -> Bayern (EPSG:32632, 10m)"):
        in_path = Path(in_tif)
        out_name = f"{in_path.stem}_10m_perennial.tif"
        out_tif = str(Path(OUT_DIR) / out_name)
        process_one(str(in_path), out_tif, bayern_geom_32632)

    print(f"\nDone. Outputs in: {OUT_DIR}")

if __name__ == "__main__":
    main()