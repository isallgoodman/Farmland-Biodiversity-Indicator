# -*- coding: utf-8 -*-gpd
"""
Clip GRASSLAND_MOW_DE_*_FIRSTCUT.tif rasters to Bayern extent and enforce EPSG:32632.
Outputs are saved with prefix: Bayern_
"""

# =========================
# 1) Imports + Settings
# =========================
import os
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from tqdm import tqdm


# =========================
# 2) Paths (EDIT IF NEEDED)
# =========================
INPUT_TIFS = [
    r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\GRASSLAND_MOW_DE_2018_FIRSTCUT.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\GRASSLAND_MOW_DE_2019_FIRSTCUT.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\GRASSLAND_MOW_DE_2020_FIRSTCUT.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\GRASSLAND_MOW_DE_2021_FIRSTCUT.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\GRASSLAND_MOW_DE_2022_FIRSTCUT.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\GRASSLAND_MOW_DE_2023_FIRSTCUT.tif",
    r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\GRASSLAND_MOW_DE_2024_FIRSTCUT.tif",
]

BAYERN_SHP = r"D:\The New Folder\MASTErARBEIT\Shapefiles\Bayern.shp"

OUT_DIR = r"D:\The New Folder\Master_arbeit\DLR_Data\Firstcut\Bayern_clipped"
OUT_PREFIX = "Bayern_"

TARGET_EPSG = 32632  # EPSG:32632


# =========================
# 3) Helpers
# =========================
def load_bayern_geometry(shp_path: str, target_epsg: int):
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError(f"No features found in: {shp_path}")

    gdf = gdf.to_crs(epsg=target_epsg)
    geom = gdf.unary_union  # dissolve to single geometry
    return [geom]  # rasterio.mask expects an iterable of geometries


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def clip_to_bayern_32632(in_tif: str, out_tif: str, bayern_geom_32632):
    with rasterio.open(in_tif) as src:
        # On-the-fly reprojection into EPSG:32632 (no intermediate files)
        with WarpedVRT(
            src,
            crs=f"EPSG:{TARGET_EPSG}",
            resampling=Resampling.nearest,
        ) as vrt:
            out_image, out_transform = mask(
                vrt,
                bayern_geom_32632,
                crop=True,
                all_touched=False,
                filled=True,
                nodata=vrt.nodata,
            )

            out_meta = vrt.meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "crs": f"EPSG:{TARGET_EPSG}",
                    "compress": "LZW",
                }
            )

            # If nodata is undefined, set a sensible default (keeps output consistent)
            if out_meta.get("nodata", None) is None:
                out_meta["nodata"] = 0

            with rasterio.open(out_tif, "w", **out_meta) as dst:
                dst.write(out_image)


# =========================
# 4) Main
# =========================
def main():
    ensure_dir(OUT_DIR)

    bayern_geom_32632 = load_bayern_geometry(BAYERN_SHP, TARGET_EPSG)

    for in_tif in tqdm(INPUT_TIFS, desc="Clipping to Bayern (EPSG:32632)"):
        in_path = Path(in_tif)
        out_name = OUT_PREFIX + in_path.name
        out_tif = str(Path(OUT_DIR) / out_name)

        clip_to_bayern_32632(str(in_path), out_tif, bayern_geom_32632)

    print(f"\nDone. Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
