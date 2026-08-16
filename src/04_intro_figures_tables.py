import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from matplotlib.patches import Circle, Polygon, Rectangle
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "intro_figures_outputs"
LAD_SHP = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.shp"
POINTS_CSV = ROOT / "outputs" / "hdbscan_retail_clusters_assignments.csv"
GRID_WIDE = ROOT / "outputs" / "grid_250m_spatial_analysis" / "grid_250m_change_metrics_wide.csv"
SCALE_STATS = ROOT / "outputs" / "remade_progress_figures" / "remade_retail_scale_floor_area_stats.csv"

CITY_ORDER = ["Birmingham", "Liverpool"]
CITY_CODES = {"Birmingham": "E08000025", "Liverpool": "E08000012"}
CITY_COLORS = {"Birmingham": "#4C78A8", "Liverpool": "#F58518"}
CITY_CENTRES_LONLAT = {
    # Indicative city-centre points used to define the focus ring on Figure 2.
    "Birmingham": (-1.9025, 52.4795),
    "Liverpool": (-2.9916, 53.4072),
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
})


def save_pub(fig, stem, dpi=600):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_lad_boundaries():
    reader = shapefile.Reader(str(LAD_SHP))
    fields = [f.name for f in reader.fields[1:]]
    boundaries = {}
    records = []
    for sr in reader.iterShapeRecords():
        rec = dict(zip(fields, sr.record))
        name = rec.get("LAD23NM")
        code = rec.get("LAD23CD")
        pts = np.array(sr.shape.points, dtype=float)
        if len(pts) == 0:
            continue
        parts = list(sr.shape.parts) + [len(pts)]
        segments = [pts[s:e] for s, e in zip(parts[:-1], parts[1:])]
        boundaries[name] = segments
        records.append({"name": name, "code": code, "segments": segments})
    return records, boundaries


def draw_segments(ax, segments, **kwargs):
    for seg in segments:
        ax.plot(seg[:, 0], seg[:, 1], **kwargs)


def draw_decoration_backplate(ax, x=0.035, y=0.025, width=0.33, height=0.24):
    ax.add_patch(Rectangle((x, y), width, height, transform=ax.transAxes,
                           facecolor="white", edgecolor="none", alpha=0.86,
                           clip_on=False, zorder=8))


def draw_north_arrow(ax, x=0.08, y=0.12, size=0.075):
    """Draw a compact cartographic north arrow in axes-fraction coordinates."""
    outer = np.array([
        [x, y + size],
        [x - size * 0.35, y - size * 0.15],
        [x, y + size * 0.18],
        [x + size * 0.35, y - size * 0.15],
    ])
    inner = np.array([
        [x, y + size * 0.82],
        [x - size * 0.10, y + size * 0.13],
        [x, y + size * 0.28],
        [x + size * 0.10, y + size * 0.13],
    ])
    ax.add_patch(Polygon(outer, closed=True, facecolor="white", edgecolor="#111111",
                         lw=0.8, transform=ax.transAxes, clip_on=False, zorder=10))
    ax.add_patch(Polygon(inner, closed=True, facecolor="#111111", edgecolor="#111111",
                         lw=0.4, transform=ax.transAxes, clip_on=False, zorder=11))


def draw_scale_bar(ax, length_m, label, loc=(0.08, 0.08), linewidth=1.8):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + (x1 - x0) * loc[0]
    y = y0 + (y1 - y0) * loc[1]
    half = length_m / 2
    tick = (y1 - y0) * 0.010
    ax.plot([x, x + length_m], [y, y], color="#222222", lw=linewidth,
            solid_capstyle="butt", zorder=12)
    for xpos in [x, x + half, x + length_m]:
        ax.plot([xpos, xpos], [y - tick, y + tick], color="#222222", lw=0.8, zorder=12)

    total_km = length_m / 1000
    half_km = total_km / 2
    def fmt(v):
        return f"{int(v)}" if abs(v - round(v)) < 1e-6 else f"{v:g}"

    text_y = y + (y1 - y0) * 0.020
    ax.text(x, text_y, "0", ha="center", va="bottom", fontsize=5.6, color="#222222", zorder=13)
    ax.text(x + half, text_y, fmt(half_km), ha="center", va="bottom",
            fontsize=5.6, color="#222222", zorder=13)
    ax.text(x + length_m, text_y, f"{fmt(total_km)} km", ha="center", va="bottom",
            fontsize=5.6, color="#222222", zorder=13)


def transform_points(lon, lat):
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    return transformer.transform(lon, lat)


def figure_uk_locator(records, boundaries):
    fig, ax = plt.subplots(figsize=(5.2, 7.0))
    for rec in records:
        # Very light national/local-authority context. No graticule/lat-long grid.
        for seg in rec["segments"]:
            ax.plot(seg[:, 0], seg[:, 1], color="#D2D7DD", lw=0.22, zorder=1)

    for city in CITY_ORDER:
        draw_segments(ax, boundaries[city], color=CITY_COLORS[city], lw=1.6, zorder=3)
        pts = np.vstack(boundaries[city])
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        ax.scatter(cx, cy, s=42, color=CITY_COLORS[city], edgecolor="white", lw=0.7, zorder=4)
        dx = 26000 if city == "Birmingham" else 22000
        dy = -8000 if city == "Birmingham" else 16000
        ax.annotate(city, xy=(cx, cy), xytext=(cx + dx, cy + dy),
                    arrowprops=dict(arrowstyle="-", color=CITY_COLORS[city], lw=0.8),
                    fontsize=8.8, fontweight="bold", color="#24313D", zorder=5)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-90000, 660000)
    ax.set_ylim(0, 1250000)
    ax.axis("off")
    draw_north_arrow(ax, 0.08, 0.13, 0.075)
    draw_scale_bar(ax, 100000, "100 km", loc=(0.09, 0.065))
    ax.set_title("Figure 1. Location of the case-study cities in the United Kingdom",
                 loc="left", fontsize=10, fontweight="bold", pad=8)
    ax.text(0.02, 0.02, "Boundary source: LAD May 2023 UK BGC; projection: British National Grid.",
            transform=ax.transAxes, fontsize=6.5, color="#6B7280")
    save_pub(fig, "figure1_uk_case_city_locator")


def figure_city_context(boundaries):
    points = pd.read_csv(POINTS_CSV)
    px, py = transform_points(points["lon"].to_numpy(), points["lat"].to_numpy())
    points["easting"] = px
    points["northing"] = py

    centres = {}
    for city, (lon, lat) in CITY_CENTRES_LONLAT.items():
        centres[city] = transform_points(np.array([lon]), np.array([lat]))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.75), constrained_layout=True)
    for ax, city in zip(axes, CITY_ORDER):
        city_points = points[points["city"].eq(city)]
        ax.scatter(city_points["easting"], city_points["northing"], s=1.0, color="#303030",
                   alpha=0.16, rasterized=True, linewidths=0, label="OpenLocal retail points")
        draw_segments(ax, boundaries[city], color="#222222", lw=1.0, zorder=4)

        cx, cy = centres[city][0][0], centres[city][1][0]
        ax.scatter(cx, cy, s=38, color=CITY_COLORS[city], edgecolor="white", lw=0.8, zorder=5)
        ax.add_patch(Circle((cx, cy), radius=2000, fill=False, lw=1.2,
                            edgecolor=CITY_COLORS[city], linestyle="--", zorder=5))
        ax.text(cx + 2300, cy + 800, "Indicative\ncity-centre\nfocus area",
                fontsize=7, color=CITY_COLORS[city], va="center")

        pts = np.vstack(boundaries[city])
        pad = 1200
        ax.set_xlim(pts[:, 0].min() - 5200, pts[:, 0].max() + pad)
        ax.set_ylim(pts[:, 1].min() - 4300, pts[:, 1].max() + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(city, fontsize=8.5, fontweight="bold", pad=1.5)
        ax.axis("off")
        draw_north_arrow(ax, 0.08, 0.16, 0.08)
        draw_scale_bar(ax, 5000, "5 km", loc=(0.08, 0.065))

    fig.text(0.02, -0.025, "Figure 2.", color="#D95F02",
             fontsize=6.8, fontweight="bold", va="bottom")
    fig.text(0.095, -0.025,
             "City boundaries, indicative city-centre focus areas and OpenLocal retail points. "
             "City-centre focus areas are indicative 2 km rings; no graticule is shown.",
             fontsize=6.8, color="#111111", va="bottom")
    save_pub(fig, "figure2_city_boundaries_centres_openlocal_points")


def table_case_city_info():
    scale = pd.read_csv(SCALE_STATS)
    grid = pd.read_csv(GRID_WIDE)
    city_values = {}
    for city in CITY_ORDER:
        city_scale = scale[scale["geocode_name"].eq(city)]
        total_records = int(city_scale["retail_units"].sum())
        records_2016 = int(city_scale.loc[city_scale["year"].eq(2016), "retail_units"].iloc[0])
        records_2025 = int(city_scale.loc[city_scale["year"].eq(2025), "retail_units"].iloc[0])
        active_cells = int(grid[grid["city"].eq(city)].shape[0])
        city_values[city] = {
            "Local authority code": CITY_CODES[city],
            "Population": "1,144,900\n(Census 2021)" if city == "Birmingham" else "486,100\n(Census 2021)",
            "Median age": "34\n(Census 2021)" if city == "Birmingham" else "35\n(Census 2021)",
            "Employment / unemployment": "Employment 62.1%;\nunemployment 7.7%\n(2023 profile)" if city == "Birmingham" else "Employment 67.5%;\nunemployment 7.0%\n(Jan-Dec 2023 profile)",
            "Study period": "2016-2025",
            "OpenLocal retail records": f"{total_records:,} total\n{records_2016:,} in 2016\n{records_2025:,} in 2025",
            "250 m grid coverage": f"{active_cells:,}\nactive grid cells",
        }

    indicators = [
        "Local authority code",
        "Population",
        "Median age",
        "Employment / unemployment",
        "Study period",
        "OpenLocal retail records",
        "250 m grid coverage",
    ]
    df = pd.DataFrame({
        "Indicator": indicators,
        "Birmingham": [city_values["Birmingham"][i] for i in indicators],
        "Liverpool": [city_values["Liverpool"][i] for i in indicators],
    })
    df.to_csv(OUT / "table1_case_city_basic_information.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7.2, 3.45))
    ax.axis("off")
    ax.text(0.00, 0.98, "Table 1.", transform=ax.transAxes,
            color="#D95F02", fontsize=7.0, fontweight="bold", va="top")
    ax.text(0.105, 0.98, "Summary statistics for selected case-study cities.",
            transform=ax.transAxes, color="#111111", fontsize=7.0, va="top")
    ax.plot([0, 1], [0.915, 0.915], transform=ax.transAxes, color="#F58518", lw=0.8)
    ax.plot([0, 1], [0.145, 0.145], transform=ax.transAxes, color="#F58518", lw=0.8)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        colLoc="left",
        colWidths=[0.31, 0.345, 0.345],
        bbox=[0.0, 0.16, 1.0, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#FFFFFF")
        cell.set_linewidth(0)
        cell.get_text().set_wrap(True)
        if r == 0:
            cell.set_facecolor("#FFFFFF")
            cell.set_text_props(weight="bold", color="#111111")
        else:
            cell.set_facecolor("#FFFFFF")
            if c == 0:
                cell.set_text_props(color="#111111", weight="normal")
    ax.text(0, 0.055,
            "Source: ONS Census 2021 area profiles; Nomis labour market profiles, 2023; OpenLocal retail property dataset. OpenLocal coverage is computed from the dissertation retail analysis dataset.",
            transform=ax.transAxes, fontsize=5.3, color="#6B7280", va="top", wrap=True)
    save_pub(fig, "table1_case_city_basic_information", dpi=300)


def case_city_table_dataframe():
    scale = pd.read_csv(SCALE_STATS)
    grid = pd.read_csv(GRID_WIDE)
    city_values = {}
    for city in CITY_ORDER:
        city_scale = scale[scale["geocode_name"].eq(city)]
        total_records = int(city_scale["retail_units"].sum())
        records_2016 = int(city_scale.loc[city_scale["year"].eq(2016), "retail_units"].iloc[0])
        records_2025 = int(city_scale.loc[city_scale["year"].eq(2025), "retail_units"].iloc[0])
        active_cells = int(grid[grid["city"].eq(city)].shape[0])
        city_values[city] = {
            "Local authority code": CITY_CODES[city],
            "Population": "1,144,900\n(Census 2021)" if city == "Birmingham" else "486,100\n(Census 2021)",
            "Median age": "34\n(Census 2021)" if city == "Birmingham" else "35\n(Census 2021)",
            "Employment /\nunemployment": "Employment 62.1%;\nunemployment 7.7%\n(2023 profile)" if city == "Birmingham" else "Employment 67.5%;\nunemployment 7.0%\n(Jan-Dec 2023)",
            "Study period": "2016-2025",
            "OpenLocal retail\nrecords": f"{total_records:,} total\n{records_2016:,} in 2016\n{records_2025:,} in 2025",
            "250 m grid\ncoverage": f"{active_cells:,}\nactive cells",
        }
    indicators = [
        "Local authority code",
        "Population",
        "Median age",
        "Employment /\nunemployment",
        "Study period",
        "OpenLocal retail\nrecords",
        "250 m grid\ncoverage",
    ]
    return pd.DataFrame({
        "Selected\ncase city": indicators,
        "Birmingham": [city_values["Birmingham"][i] for i in indicators],
        "Liverpool": [city_values["Liverpool"][i] for i in indicators],
    })


def draw_city_panel(ax, boundaries, points, city):
    city_points = points[points["city"].eq(city)]
    ax.scatter(city_points["easting"], city_points["northing"], s=0.8, color="#303030",
               alpha=0.16, rasterized=True, linewidths=0, zorder=1)
    draw_segments(ax, boundaries[city], color="#303030", lw=0.85, zorder=3)

    cx_arr, cy_arr = transform_points(
        np.array([CITY_CENTRES_LONLAT[city][0]]),
        np.array([CITY_CENTRES_LONLAT[city][1]]),
    )
    cx, cy = cx_arr[0], cy_arr[0]
    ax.scatter(cx, cy, s=24, color=CITY_COLORS[city], edgecolor="white", lw=0.65, zorder=4)
    ax.add_patch(Circle((cx, cy), radius=2000, fill=False, lw=1.0,
                        edgecolor=CITY_COLORS[city], linestyle=(0, (4, 2)), zorder=4))

    pts = np.vstack(boundaries[city])
    pad = 1400
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.text(0.02, 0.96, city, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.2, fontweight="bold", color="#222222")
    ax.text(0.02, 0.89, "OpenLocal retail points and indicative 2 km city-centre ring",
            transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color="#555555")
    draw_north_arrow(ax, 0.08, 0.16, 0.075)
    draw_scale_bar(ax, 5000, "5 km", loc=(0.06, 0.075), linewidth=1.8)


def figure_intro_plate_like_example(boundaries):
    points = pd.read_csv(POINTS_CSV)
    px, py = transform_points(points["lon"].to_numpy(), points["lat"].to_numpy())
    points["easting"] = px
    points["northing"] = py
    table_df = case_city_table_dataframe()

    fig = plt.figure(figsize=(12.4, 5.1))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.10, 1.18],
        height_ratios=[1, 1],
        left=0.035, right=0.985, top=0.93, bottom=0.14,
        wspace=0.10, hspace=0.12,
    )
    ax_bham = fig.add_subplot(gs[0, 0])
    ax_liv = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[:, 1])

    draw_city_panel(ax_bham, boundaries, points, "Birmingham")
    draw_city_panel(ax_liv, boundaries, points, "Liverpool")

    ax_table.axis("off")
    ax_table.text(0.00, 0.89, "Table 1.", transform=ax_table.transAxes,
                  color="#D95F02", fontsize=7.2, fontweight="bold", va="center")
    ax_table.text(0.085, 0.89, "Summary statistics for selected case-study cities.",
                  transform=ax_table.transAxes, color="#111111", fontsize=7.2, va="center")
    ax_table.plot([0, 1], [0.865, 0.865], transform=ax_table.transAxes, color="#F58518", lw=0.8)
    ax_table.plot([0, 1], [0.175, 0.175], transform=ax_table.transAxes, color="#F58518", lw=0.8)

    table = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        colLoc="center",
        colWidths=[0.36, 0.32, 0.32],
        bbox=[0.00, 0.19, 1.00, 0.66],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.7)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#FFFFFF")
        cell.set_linewidth(0)
        cell.get_text().set_wrap(True)
        if r == 0:
            cell.set_text_props(weight="bold", color="#111111")
            cell.set_facecolor("#FFFFFF")
        else:
            cell.set_facecolor("#FFFFFF")
            if c == 0:
                cell.set_text_props(ha="left", color="#222222")
            else:
                cell.set_text_props(color="#222222")

    ax_table.text(0.00, 0.105,
                  "Source: ONS Census 2021 area profiles; Nomis labour market profiles, 2023; OpenLocal retail property dataset. "
                  "OpenLocal coverage is computed from the dissertation retail analysis dataset.",
                  transform=ax_table.transAxes, fontsize=5.6, color="#5F6670", va="top", wrap=True)

    fig.text(0.035, 0.055, "Figure 1.", color="#D95F02", fontsize=7.0, fontweight="bold", va="bottom")
    fig.text(0.088, 0.055,
             "Case-study city boundaries, city-centre focus areas and OpenLocal retail data coverage for Birmingham and Liverpool. "
             "Maps use British National Grid; no latitude/longitude graticule is shown.",
             color="#111111", fontsize=7.0, va="bottom")
    save_pub(fig, "figure1_case_study_plate_map_table_style")


def write_sources_note():
    text = """# Introductory figures and Table 1 notes

Generated with Python from:

- LAD May 2023 UK BGC boundary shapefile.
- OpenLocal-derived retail point assignments and 250 m grid outputs produced earlier in this workflow.
- ONS Census 2021 area profiles and Nomis labour market profiles for contextual case-study indicators.

Important:

- Figure 2 uses indicative 2 km city-centre focus rings around central reference points. Replace these with official city-centre / BID / planning boundaries if those become available.
- Table 1 demographic/labour-market values should use the final citation format adopted in the dissertation reference list.
"""
    (OUT / "README_intro_figures_table.md").write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, boundaries = load_lad_boundaries()
    figure_uk_locator(records, boundaries)
    figure_city_context(boundaries)
    table_case_city_info()
    write_sources_note()


if __name__ == "__main__":
    main()

