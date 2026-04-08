# FHBI_hexagon.py
# Compute Farmland Habitat Biodiversity Indicator (FHBI) per hexagon
# using area-share weighted mean of raster classes (9-class scheme).
#
# INPUTS:
#   Raster:  D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled\mosaic_max_all_layers_BayernExtent.tif
#   Hexagons: D:\The New Folder\Master_arbeit\DLR_Data\Shapefiles\Hex_farmland.gpkg
#
# OUTPUT:
#   D:\The New Folder\Master_arbeit\DLR_Data\FHBI_hexagon.gpkg
#   -> adds column "FHBI"

from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
from tqdm import tqdm



# Paths

RASTER_PATH = r"D:\The New Folder\Master_arbeit\DLR_Data\Intermediates\Intermediates_resampled\Habitat_stitched.tif"
HEX_PATH = r"D:\The New Folder\Master_arbeit\DLR_Data\Shapefiles\Hex_farmland.gpkg"
OUT_GPKG = r"D:\The New Folder\Master_arbeit\DLR_Data\FHBI_hexagon.gpkg"



# Class -> weight mapping

# IMPORTANT:
# Set these keys to the actual raster codes in your TIFF.
#
# Common cases:
#   A) raster codes are 1..9 for:
#      1=Very low, 2=Very low-low, 3=Low, 4=Low-medium, 5=Medium,
#      6=Medium-high, 7=High, 8=High-very high, 9=Very high
#
# If your raster codes differ, edit the keys below.
CLASS_TO_WEIGHT = {
    1: 0.000,
    2: 0.125,
    3: 0.250,
    4: 0.375,
    5: 0.500,
    6: 0.625,
    7: 0.750,
    8: 0.875,
    9: 1.000,
}


def compute_fhbi_for_geom(src, geom, class_to_weight):
    """
    Returns FHBI (float) for a single polygon geometry based on raster class shares.
    """
    # bounds -> window
    minx, miny, maxx, maxy = geom.bounds
    win = from_bounds(minx, miny, maxx, maxy, transform=src.transform)

    # Guard against empty/invalid windows
    if win.width <= 0 or win.height <= 0:
        return np.nan

    arr = src.read(1, window=win, masked=False)
    if arr.size == 0:
        return np.nan

    # Build mask of pixels inside polygon (True=inside)
    # geometry_mask returns True for pixels OUTSIDE shapes by default -> invert=True makes inside=True
    win_transform = src.window_transform(win)
    inside = geometry_mask(
        [geom],
        out_shape=arr.shape,
        transform=win_transform,
        invert=True,
        all_touched=False,  # change to True if you want to count edge pixels
    )

    if not inside.any():
        return np.nan

    vals = arr[inside]

    # Drop nodata if defined
    if src.nodata is not None:
        vals = vals[vals != src.nodata]
        if vals.size == 0:
            return np.nan

    # If raster already stores weights in [0,1] (float), FHBI is just the mean of those values

    if np.issubdtype(vals.dtype, np.floating):
        finite = np.isfinite(vals)
        if not finite.any():
            return np.nan
        v = vals[finite]
        # If values look like weights (0..1), use mean directly
        if v.min() >= -1e-6 and v.max() <= 1.0 + 1e-6:
            return float(v.mean())

    # Otherwise: treat as categorical classes and apply weighted share formula.
    # Count only classes present in mapping; ignore others (including 0 background etc.)
    total = 0
    score = 0.0

    unique, counts = np.unique(vals, return_counts=True)
    for cls, cnt in zip(unique.tolist(), counts.tolist()):
        if cls in class_to_weight:
            total += cnt
            score += cnt * float(class_to_weight[cls])

    if total == 0:
        return np.nan

    return score / total  # 0..1


def main():
    raster_path = Path(RASTER_PATH)
    hex_path = Path(HEX_PATH)
    out_path = Path(OUT_GPKG)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(hex_path)

    with rasterio.open(raster_path) as src:
        # Reproject hexagons to raster CRS if needed
        if gdf.crs is None:
            raise ValueError("Hexagon file has no CRS. Please assign a CRS before running.")
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        fhbi_vals = []
        for geom in tqdm(gdf.geometry, total=len(gdf), desc="Computing FHBI per hexagon"):
            if geom is None or geom.is_empty:
                fhbi_vals.append(np.nan)
                continue
            fhbi_vals.append(compute_fhbi_for_geom(src, geom, CLASS_TO_WEIGHT))

    gdf["FHBI"] = fhbi_vals

    # Save as a single-layer GeoPackage

    gdf.to_file(out_path, driver="GPKG")

    print(f"\nDone.\nSaved: {out_path}\nColumn added: FHBI")


if __name__ == "__main__":
    main()