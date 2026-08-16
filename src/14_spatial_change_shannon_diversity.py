from pathlib import Path
import sys

import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "outputs" / "grid_250m_spatial_analysis"
WIDE_CSV = OUT / "grid_250m_change_metrics_wide.csv"

sys.path.insert(0, str(WORK))
from make_spatial_change_floor_area_vertical import (  # noqa: E402
    GRID_SIZE,
    SCALE,
    W,
    H,
    CITY_ORDER,
    PAL,
    text,
    line,
    rect,
    save_all,
    load_boundaries,
    data_extent,
    transform_for_extent,
    draw_boundary,
    draw_north,
    draw_scale,
)


VMIN, VMAX = -0.7, 0.7
NEG = (208, 72, 153)
MID = (232, 234, 233)
POS = (76, 175, 74)


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def color_for_value(value):
    if pd.isna(value):
        return None
    v = max(VMIN, min(VMAX, float(value)))
    if v < 0:
        t = (v - VMIN) / (0 - VMIN)
        return rgb_to_hex(mix(NEG, MID, t))
    t = v / VMAX
    return rgb_to_hex(mix(MID, POS, t))


def draw_cells(draw, cells, tr):
    cells = cells[pd.notna(cells["business_diversity_change"])].copy()
    cells["_abs"] = cells["business_diversity_change"].abs()
    for row in cells.sort_values("_abs").itertuples():
        color = color_for_value(row.business_diversity_change)
        if color is None:
            continue
        x1, y1 = tr(row.grid_x, row.grid_y + GRID_SIZE)
        x2, y2 = tr(row.grid_x + GRID_SIZE, row.grid_y)
        rect(draw, (x1, y1, x2, y2), color, outline="#FFFFFF", width=0.5)


def draw_gradient_legend(draw, x, y):
    text(draw, (x, y), "Change in Shannon", size=25, bold=True)
    text(draw, (x, y + 34), "diversity, 2016-2025", size=25, bold=True)
    bar_x, bar_y = x, y + 78
    bar_w, bar_h = 44, 215
    for i in range(bar_h):
        value = VMAX - (i / (bar_h - 1)) * (VMAX - VMIN)
        color = color_for_value(value)
        rect(draw, (bar_x, bar_y + i, bar_x + bar_w, bar_y + i + 1), color)
    rect(draw, (bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), fill=None, outline="#BFC8CE", width=1)

    for value in [0.6, 0.3, 0.0, -0.3, -0.6]:
        yy = bar_y + (VMAX - value) / (VMAX - VMIN) * bar_h
        line(draw, [(bar_x + bar_w, yy), (bar_x + bar_w + 11, yy)], PAL["ink"], width=1.5)
        text(draw, (bar_x + bar_w + 20, yy - 13), f"{value:.1f}", size=20)

    yy = bar_y + bar_h + 54
    line(draw, [(x, yy), (x + 76, yy)], PAL["boundary"], width=1.5)
    text(draw, (x + 92, yy - 17), "Local authority boundary", size=22)


def main():
    data = pd.read_csv(WIDE_CSV)
    boundaries = load_boundaries()
    panels = {
        "Birmingham": (55, 105, 1540, 1010),
        "Liverpool": (55, 1260, 1540, 1010),
    }
    extents = {}
    common_scale = None
    for city, panel in panels.items():
        cells = data[data["city"].eq(city)].copy()
        boundary = boundaries.get(city, [])
        extent = data_extent(cells, boundary)
        extents[city] = extent
        xmin, xmax, ymin, ymax = extent
        city_scale = min(panel[2] / (xmax - xmin), panel[3] / (ymax - ymin))
        common_scale = city_scale if common_scale is None else min(common_scale, city_scale)

    img = Image.new("RGBA", (W * SCALE, H * SCALE), PAL["bg"])
    draw = ImageDraw.Draw(img)

    for city, panel in panels.items():
        cells = data[data["city"].eq(city)].copy()
        boundary = boundaries.get(city, [])
        tr, panel_scale = transform_for_extent(extents[city], panel, fixed_scale=common_scale, align_x=0.31)
        rect(draw, (panel[0], panel[1], panel[0] + panel[2], panel[1] + panel[3]), PAL["panel_bg"], outline=PAL["panel_line"], width=1.4)
        text(draw, (panel[0] + panel[2] / 2, panel[1] - 38), city, size=40, bold=True, anchor="mm")
        draw_cells(draw, cells, tr)
        draw_boundary(draw, boundary, tr)
        draw_north(draw, panel[0] + 72, panel[1] + 88, size=48)
        draw_scale(draw, panel[0] + 58, panel[1] + panel[3] - 70, panel_scale, length_m=3000)
        draw_gradient_legend(draw, panel[0] + panel[2] - 420, panel[1] + panel[3] - 470)
    save_all(img, "figure_spatial_change_business_diversity_vertical")

    single_panel = (75, 120, 2260, 1400)
    for city in CITY_ORDER:
        single_img = Image.new("RGBA", (2500 * SCALE, 1700 * SCALE), PAL["bg"])
        single_draw = ImageDraw.Draw(single_img)
        cells = data[data["city"].eq(city)].copy()
        boundary = boundaries.get(city, [])
        tr, panel_scale = transform_for_extent(extents[city], single_panel, fixed_scale=None, align_x=0.31)
        rect(
            single_draw,
            (
                single_panel[0],
                single_panel[1],
                single_panel[0] + single_panel[2],
                single_panel[1] + single_panel[3],
            ),
            PAL["panel_bg"],
            outline=PAL["panel_line"],
            width=1.4,
        )
        text(single_draw, (single_panel[0] + single_panel[2] / 2, single_panel[1] - 50), city, size=58, bold=True, anchor="mm")
        draw_cells(single_draw, cells, tr)
        draw_boundary(single_draw, boundary, tr)
        draw_north(single_draw, single_panel[0] + 105, single_panel[1] + 130, size=64)
        draw_scale(single_draw, single_panel[0] + 90, single_panel[1] + single_panel[3] - 105, panel_scale, length_m=3000)
        draw_gradient_legend(single_draw, single_panel[0] + single_panel[2] - 650, single_panel[1] + single_panel[3] - 675)
        save_all(single_img, f"figure_spatial_change_business_diversity_{city.lower()}")


if __name__ == "__main__":
    main()

