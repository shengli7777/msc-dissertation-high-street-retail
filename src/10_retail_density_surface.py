from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs" / "grid_250m_spatial_analysis"
sys.path.insert(0, str(WORK))

import make_business_count_indicator_map_2016_2025 as base


YEARS = [2016, 2019, 2022, 2025]
CITIES = ["Birmingham", "Liverpool"]
GRID_SIZE = 250
SIGMA_CELLS = 2.2
DENSITY_THRESHOLD = 0.08

SCALE = 2
W, H = 1580, 1720

PAL = {
    "bg": "#FFFFFF",
    "ink": "#25313A",
    "boundary": "#7A8791",
    "panel_bg": "#FBFCFD",
    "panel_line": "#E6EEF2",
    "dark": "#17212B",
}

LEVELS = [
    (0.08, 0.5, "Very low", "#EEF5F2"),
    (0.5, 1.5, "Low", "#CDE5DC"),
    (1.5, 3.5, "Moderate", "#94C9BD"),
    (3.5, 8, "High", "#55A8A0"),
    (8, 18, "Very high", "#257D8A"),
    (18, math.inf, "Highest", "#4C3F7D"),
]


def density_color(value):
    if value < DENSITY_THRESHOLD or not np.isfinite(value):
        return None
    for low, high, _, color in LEVELS:
        if low <= value < high:
            return color
    return LEVELS[-1][3]


def point_in_ring(x, y, ring):
    inside = False
    if len(ring) < 3:
        return False
    x0, y0 = ring[-1]
    for x1, y1 in ring:
        crosses = ((y1 > y) != (y0 > y)) and (x < (x0 - x1) * (y - y1) / ((y0 - y1) or 1e-9) + x1)
        if crosses:
            inside = not inside
        x0, y0 = x1, y1
    return inside


def inside_boundary(x, y, boundary):
    return any(point_in_ring(x, y, ring) for ring in boundary)


def gaussian_kernel(sigma_cells=2.2):
    radius = int(math.ceil(sigma_cells * 4))
    xs = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(xs ** 2) / (2 * sigma_cells ** 2))
    kernel /= kernel.sum()
    return kernel


def smooth_grid(city_grid, year, extent, boundary):
    xmin, xmax, ymin, ymax = extent
    x_vals = np.arange(math.floor(xmin / GRID_SIZE) * GRID_SIZE, math.ceil(xmax / GRID_SIZE) * GRID_SIZE + GRID_SIZE, GRID_SIZE)
    y_vals = np.arange(math.floor(ymin / GRID_SIZE) * GRID_SIZE, math.ceil(ymax / GRID_SIZE) * GRID_SIZE + GRID_SIZE, GRID_SIZE)
    z = np.zeros((len(y_vals), len(x_vals)), dtype=float)

    x_lookup = {int(x): i for i, x in enumerate(x_vals)}
    y_lookup = {int(y): i for i, y in enumerate(y_vals)}
    sub = city_grid[city_grid["year"].eq(year)]
    for row in sub.itertuples():
        ix = x_lookup.get(int(row.grid_x))
        iy = y_lookup.get(int(row.grid_y))
        if ix is not None and iy is not None:
            z[iy, ix] += float(row.retail_count)

    kernel = gaussian_kernel(SIGMA_CELLS)
    z = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=z)
    z = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=z)

    mask = np.zeros_like(z, dtype=bool)
    for iy, gy in enumerate(y_vals):
        cy = gy + GRID_SIZE / 2
        for ix, gx in enumerate(x_vals):
            cx = gx + GRID_SIZE / 2
            mask[iy, ix] = inside_boundary(cx, cy, boundary)
    z[~mask] = np.nan
    return x_vals, y_vals, z


def draw_density_cells(draw, x_vals, y_vals, density, tr):
    rows, cols = density.shape
    for iy in range(rows):
        for ix in range(cols):
            color = density_color(density[iy, ix])
            if color is None:
                continue
            gx = x_vals[ix]
            gy = y_vals[iy]
            x1, y1 = tr(gx, gy + GRID_SIZE)
            x2, y2 = tr(gx + GRID_SIZE, gy)
            base.rect(draw, (x1, y1, x2, y2), color, outline=None)


def draw_legend(draw, x, y):
    base.text(draw, (x, y), "Smoothed retail-record", size=18, bold=True)
    base.text(draw, (x, y + 21), "density surface", size=18, bold=True)
    for i, (_, _, label, color) in enumerate(LEVELS):
        yy = y + 56 + i * 31
        base.rect(draw, (x, yy - 11, x + 24, yy + 11), color, outline="#FFFFFF", width=1)
        base.text(draw, (x + 34, yy - 12), label, size=15)
    yy = y + 256
    base.line(draw, [(x, yy), (x + 48, yy)], PAL["boundary"], width=2)
    base.text(draw, (x + 60, yy - 12), "Local authority boundary", size=15)
    base.text(draw, (x, yy + 38), "Bandwidth: 550 m", size=13, fill="#65727A")


def main():
    city_refs = {}
    city_grids = {}
    for city in CITIES:
        _, lon0, lat0 = base.load_city_points(city)
        city_refs[city] = (lon0, lat0)
        city_grids[city] = pd.read_csv(OUT / f"{city.lower()}_250m_retail_count_selected_years.csv")

    boundaries = base.load_boundaries(city_refs)
    extents = {city: base.data_extent(city_grids[city], boundaries[city]) for city in CITIES}

    img = Image.new("RGBA", (W * SCALE, H * SCALE), PAL["bg"])
    draw = ImageDraw.Draw(img)

    x_positions = {"Birmingham": 120, "Liverpool": 695}
    row_tops = {year: 105 + idx * 390 for idx, year in enumerate(YEARS)}
    panel_w, panel_h = 545, 365
    scale_ref = None

    base.text(draw, (x_positions["Birmingham"] + panel_w / 2, 55), "Birmingham", size=27, bold=True, anchor="mm")
    base.text(draw, (x_positions["Liverpool"] + panel_w / 2, 55), "Liverpool", size=27, bold=True, anchor="mm")

    for year in YEARS:
        y0 = row_tops[year]
        base.text(draw, (75, y0 + panel_h / 2), str(year), size=24, bold=True, anchor="mm")
        for city in CITIES:
            panel = (x_positions[city], y0, panel_w, panel_h)
            base.rect(draw, (panel[0], panel[1], panel[0] + panel[2], panel[1] + panel[3]), PAL["panel_bg"], outline=PAL["panel_line"])
            tr, s = base.transform_for_extent(extents[city], panel)
            if city == "Birmingham" and year == 2025:
                scale_ref = s
            x_vals, y_vals, density = smooth_grid(city_grids[city], year, extents[city], boundaries[city])
            draw_density_cells(draw, x_vals, y_vals, density, tr)
            base.draw_boundary(draw, boundaries[city], tr)

    base.draw_north(draw, 50, 80, size=34)
    base.draw_scale(draw, 52, 1665, scale_ref, length_m=6000)
    draw_legend(draw, 1265, 1350)
    base.save_all(img, "figure_retail_record_density_surface_2016_2025")


if __name__ == "__main__":
    main()

