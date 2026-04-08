# -*- coding: utf-8 -*-
"""
Clip GRASSLAND_MOW_DE_*_FREQUENCY.tif rasters to Bayern extent, EPSG:32632,
and resample to 10 m x 10 m.
Outputs are saved with prefix: Bayern_
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
from rasterio.warp import reproject, calculate_default_transform
from tqdm import tqdm


# =========================
# 2) Paths (EDIT IF NEEDED)
# =========================
INPUT_TIFS = [
    r"D:\The New Folder\Master_arbeit\DLR_Data\MF\GRASSLAND_MOW_DE_2018_FREQUENCY.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\MF\GRASSLAND_MOW_DE_2019_FREQUENCY.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\MF\GRASSLAND_MOW_DE_2020_FREQUENCY.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\MF\GRASSLAND_MOW_DE_2021_FREQUENCY.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\MF\GRASSLAND_MOW_DE_2022_FREQUENCY.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\MF\GRASSLAND_MOW_DE_2023_FREQUENCY.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\MF\GRASSLAND_MOW_DE_2024_FREQUENCY.tif",
]

BAYERN_SHP = r"D:\The New Folder\MASTErARBEIT\Shapefiles\Bayern.shp"

OUT_DIR = r"D:\The New Folder\Master_arbeit\DLR_Data\MF\Bayern_MF"
OUT_PREFIX = "Bayern_"

TARGET_EPSG = 32632          # EPSG:32632
TARGET_RES = 10              # meters (10 m x 10 m)


# =========================
# 3) Helpers
# =========================
def load_bayern_geometry(shp_path: str, target_epsg: int):
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"No features found in: {shp_path}")
    gdf = gdf.to_crs(epsg=target_epsg)
    geom = gdf.geometry.union_all()
    return [geom]


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def clip_reproject_resample_to_10m(in_tif: str, out_tif: str, bayern_geom_32632):
    """
    1) Reproject+resample raster to EPSG:32632 @ 10m using WarpedVRT
    2) Clip to Bayern
    3) Write output
    """
    with rasterio.open(in_tif) as src:
        # Build a 10m grid in EPSG:32632 covering the source bounds
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,
            f"EPSG:{TARGET_EPSG}",
            src.width,
            src.height,
            *src.bounds,
            resolution=(TARGET_RES, TARGET_RES),
        )

        nodata = src.nodata if src.nodata is not None else 0

        # VRT that outputs EPSG:32632 with 10m pixels
        with WarpedVRT(
            src,
            crs=f"EPSG:{TARGET_EPSG}",
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            resampling=Resampling.nearest,  # frequency is categorical/discrete
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
    bayern_geom_32632 = load_bayern_geometry(BAYERN_SHP, TARGET_EPSG)

    for in_tif in tqdm(INPUT_TIFS, desc="Clipping MF to Bayern (EPSG:32632, 10m)"):
        in_path = Path(in_tif)
        out_name = OUT_PREFIX + in_path.stem + "_10m.tif"
        out_tif = str(Path(OUT_DIR) / out_name)

        clip_reproject_resample_to_10m(str(in_path), out_tif, bayern_geom_32632)

    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
