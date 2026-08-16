import math
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapefile
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from pyproj import Transformer


INPUT = ROOT / "data" / "raw" / "openlocal_retail_property.parquet"
LAD_SHP = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.shp"
OUT = ROOT / "outputs" / "grid_250m_spatial_analysis"
GRID_SIZE = 250

COLS = [
    "period",
    "geocode_name",
    "uarn",
    "occupation_state",
    "category_group",
    "category_subgroup",
    "geometry",
    "rateable_value",
    "total_floor_area",
]
CITY_ORDER = ["Birmingham", "Liverpool"]
CITY_CODES = {"Birmingham": "E08000025", "Liverpool": "E08000012"}


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})


def parse_wkb_point(hex_wkb):
    if not isinstance(hex_wkb, str) or len(hex_wkb) < 42:
        return np.nan, np.nan
    raw = bytes.fromhex(hex_wkb)
    endian = "<" if raw[0] == 1 else ">"
    geom_type = struct.unpack(endian + "I", raw[1:5])[0]
    if geom_type != 1:
        return np.nan, np.nan
    return struct.unpack(endian + "dd", raw[5:21])


def shannon(values):
    counts = values.value_counts(dropna=True)
    counts = counts[counts > 0].astype(float)
    if counts.sum() <= 0:
        return np.nan
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def load_retail_points():
    df = pq.read_table(INPUT, columns=COLS).to_pandas()
    df["period"] = pd.to_datetime(df["period"])
    df["year"] = df["period"].dt.year
    df = df[df["year"].isin([2016, 2025])].copy()
    df = df[df["geocode_name"].isin(CITY_ORDER)].copy()
    df = df[df["category_group"].eq("RETAIL")].copy()
    xy = np.array([parse_wkb_point(g) for g in df["geometry"]], dtype=float)
    df["lon"] = xy[:, 0]
    df["lat"] = xy[:, 1]
    df = df[np.isfinite(df["lon"]) & np.isfinite(df["lat"])].copy()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    df["easting"], df["northing"] = transformer.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["grid_x"] = np.floor(df["easting"] / GRID_SIZE).astype(int) * GRID_SIZE
    df["grid_y"] = np.floor(df["northing"] / GRID_SIZE).astype(int) * GRID_SIZE
    df["grid_id"] = df["geocode_name"] + "_" + df["grid_x"].astype(str) + "_" + df["grid_y"].astype(str)
    df["occupation_clean"] = df["occupation_state"].fillna("UNKNOWN")
    df.loc[~df["occupation_clean"].isin(["OCCUPIED", "VACANT"]), "occupation_clean"] = "UNKNOWN"
    return df


def load_boundaries_bng():
    reader = shapefile.Reader(str(LAD_SHP))
    fields = [f.name for f in reader.fields[1:]]
    boundaries = {}
    for sr in reader.iterShapeRecords():
        rec = dict(zip(fields, sr.record))
        city = rec.get("LAD23NM")
        if city not in CITY_ORDER:
            continue
        pts = np.array(sr.shape.points, dtype=float)
        parts = list(sr.shape.parts) + [len(pts)]
        boundaries[city] = [pts[s:e] for s, e in zip(parts[:-1], parts[1:])]
    return boundaries


def aggregate_grid(df):
    rows = []
    grouped = df.groupby(["geocode_name", "year", "grid_x", "grid_y", "grid_id"], observed=True)
    for (city, year, gx, gy, gid), g in grouped:
        occupied = int((g["occupation_clean"] == "OCCUPIED").sum())
        vacant = int((g["occupation_clean"] == "VACANT").sum())
        known = occupied + vacant
        rv = g["rateable_value"].dropna()
        rows.append({
            "city": city,
            "year": int(year),
            "grid_x": int(gx),
            "grid_y": int(gy),
            "grid_id": gid,
            "business_count": int(len(g)),
            "occupied_count": occupied,
            "vacant_count": vacant,
            "known_occupation_count": known,
            "unknown_occupation_count": int((g["occupation_clean"] == "UNKNOWN").sum()),
            "vacancy_rate": float(vacant / known) if known else np.nan,
            "business_diversity": shannon(g["category_subgroup"]),
            "rateable_value_total": float(rv.sum()) if len(rv) else np.nan,
            "rateable_value_median": float(rv.median()) if len(rv) else np.nan,
            "floor_area_total": float(g["total_floor_area"].dropna().sum()) if g["total_floor_area"].notna().any() else np.nan,
        })
    grid = pd.DataFrame(rows)

    base_cols = ["city", "grid_id", "grid_x", "grid_y"]
    all_cells = grid[base_cols].drop_duplicates()
    wide_parts = []
    metrics = [
        "business_count",
        "vacancy_rate",
        "business_diversity",
        "rateable_value_total",
        "rateable_value_median",
        "known_occupation_count",
    ]
    for year in [2016, 2025]:
        sub = grid[grid["year"].eq(year)][base_cols + metrics].copy()
        sub = sub.rename(columns={m: f"{m}_{year}" for m in metrics})
        wide_parts.append(sub)
    wide = all_cells
    for part in wide_parts:
        wide = wide.merge(part, on=base_cols, how="left")
    for col in ["business_count_2016", "business_count_2025"]:
        wide[col] = wide[col].fillna(0)
    wide["business_count_change"] = wide["business_count_2025"] - wide["business_count_2016"]
    wide["business_count_pct_change"] = np.where(
        wide["business_count_2016"] > 0,
        wide["business_count_change"] / wide["business_count_2016"] * 100,
        np.nan,
    )
    wide["vacancy_rate_pp_change"] = (wide["vacancy_rate_2025"] - wide["vacancy_rate_2016"]) * 100
    wide["business_diversity_change"] = wide["business_diversity_2025"] - wide["business_diversity_2016"]
    wide["rateable_value_total_change"] = wide["rateable_value_total_2025"] - wide["rateable_value_total_2016"]
    wide["rateable_value_total_pct_change"] = np.where(
        wide["rateable_value_total_2016"] > 0,
        wide["rateable_value_total_change"] / wide["rateable_value_total_2016"] * 100,
        np.nan,
    )
    wide["cell_status"] = "persistent"
    wide.loc[(wide["business_count_2016"] > 0) & (wide["business_count_2025"] == 0), "cell_status"] = "disappeared"
    wide.loc[(wide["business_count_2016"] == 0) & (wide["business_count_2025"] > 0), "cell_status"] = "emerged"
    wide.loc[(wide["business_count_2016"] > 0) & (wide["business_count_2025"] > wide["business_count_2016"]), "cell_status"] = "intensified"
    wide.loc[(wide["business_count_2016"] > 0) & (wide["business_count_2025"] > 0) & (wide["business_count_2025"] < wide["business_count_2016"]), "cell_status"] = "declined"
    return grid, wide


def save_fig(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ["svg", "pdf"]:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def draw_boundary(ax, boundaries, city):
    for seg in boundaries.get(city, []):
        ax.plot(seg[:, 0], seg[:, 1], color="#2f2f2f", lw=0.8, alpha=0.9, zorder=3)


def add_grid_layer(ax, cells, value_col, cmap, vmin=None, vmax=None, norm=None):
    plot = cells[np.isfinite(cells[value_col])].copy()
    patches = [Rectangle((r.grid_x, r.grid_y), GRID_SIZE, GRID_SIZE) for r in plot.itertuples()]
    pc = PatchCollection(patches, cmap=cmap, linewidths=0, alpha=0.9, rasterized=True)
    pc.set_array(plot[value_col].to_numpy(dtype=float))
    if norm is not None:
        pc.set_norm(norm)
    else:
        pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    return pc


def set_city_extent(ax, boundaries, city, pad=900):
    pts = np.vstack(boundaries[city])
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("British National Grid easting (m)")
    ax.set_ylabel("British National Grid northing (m)")


def robust_limit(series, q=0.98):
    vals = series.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return 1
    lim = np.nanquantile(np.abs(vals), q)
    return float(lim if lim > 0 else 1)


def figure_grid_change(wide, boundaries):
    specs = [
        ("business_count_change", "Business count change", "RdBu", None),
        ("vacancy_rate_pp_change", "Vacancy-rate change (pp)", "RdBu_r", None),
        ("business_diversity_change", "Business diversity change", "PiYG", None),
        ("rateable_value_total_pct_change", "Rateable value change (%)", "RdBu", None),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 10.0), constrained_layout=True)
    for r, (col, label, cmap, _) in enumerate(specs):
        lim = robust_limit(wide[col])
        if col == "vacancy_rate_pp_change":
            lim = min(lim, 60)
        if col == "rateable_value_total_pct_change":
            lim = min(lim, 200)
        for c, city in enumerate(CITY_ORDER):
            ax = axes[r, c]
            city_cells = wide[wide["city"].eq(city)]
            pc = add_grid_layer(ax, city_cells, col, cmap=cmap, vmin=-lim, vmax=lim)
            draw_boundary(ax, boundaries, city)
            set_city_extent(ax, boundaries, city)
            ax.set_title(f"{city}: {label}")
            cbar = fig.colorbar(pc, ax=ax, fraction=0.044, pad=0.02)
            cbar.set_label(label)
    fig.suptitle("250 m grid-based change in High Street retail activity, 2016-2025", x=0.02, y=1.01, ha="left", fontsize=10, fontweight="bold")
    save_fig(fig, "fig_grid_250m_change_metrics")


def figure_grid_status(wide, boundaries):
    colors = {
        "disappeared": "#A95050",
        "declined": "#E7A6A1",
        "persistent": "#BDBDBD",
        "intensified": "#9BC18E",
        "emerged": "#D9A441",
    }
    order = ["disappeared", "declined", "persistent", "intensified", "emerged"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), constrained_layout=True)
    for ax, city in zip(axes, CITY_ORDER):
        cells = wide[wide["city"].eq(city)]
        for status in order:
            sub = cells[cells["cell_status"].eq(status)]
            patches = [Rectangle((r.grid_x, r.grid_y), GRID_SIZE, GRID_SIZE) for r in sub.itertuples()]
            pc = PatchCollection(patches, facecolor=colors[status], edgecolor="none", alpha=0.9, rasterized=True, label=status)
            ax.add_collection(pc)
        draw_boundary(ax, boundaries, city)
        set_city_extent(ax, boundaries, city)
        ax.set_title(city)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Grid-cell status")
    fig.suptitle("250 m grid-cell status based on business-count change", x=0.02, y=1.02, ha="left", fontsize=10, fontweight="bold")
    save_fig(fig, "fig_grid_250m_cell_status")


def figure_grid_summary(wide):
    summary = wide.groupby(["city", "cell_status"], observed=True).size().rename("n_cells").reset_index()
    summary.to_csv(OUT / "grid_250m_cell_status_summary.csv", index=False)
    order = ["disappeared", "declined", "persistent", "intensified", "emerged"]
    colors = {
        "disappeared": "#A95050",
        "declined": "#E7A6A1",
        "persistent": "#BDBDBD",
        "intensified": "#9BC18E",
        "emerged": "#D9A441",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    pivot = summary.pivot_table(index="city", columns="cell_status", values="n_cells", fill_value=0).reindex(CITY_ORDER).reindex(columns=order, fill_value=0)
    bottom = np.zeros(len(pivot.index))
    for status in order:
        axes[0].bar(pivot.index, pivot[status], bottom=bottom, color=colors[status], label=status)
        bottom += pivot[status].to_numpy()
    axes[0].set_ylabel("Number of active 250 m cells")
    axes[0].set_title("Cell status count")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plot = wide[(wide["business_count_2016"] > 0) & (wide["business_count_2025"] > 0)].copy()
    for city, color in [("Birmingham", "#4C78A8"), ("Liverpool", "#F58518")]:
        sub = plot[plot["city"].eq(city)]
        axes[1].scatter(
            sub["business_count_change"],
            sub["business_diversity_change"],
            s=np.clip(sub["business_count_2025"] * 3, 8, 80),
            alpha=0.45,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            label=city,
        )
    axes[1].axhline(0, color="#808080", lw=0.8, ls="--")
    axes[1].axvline(0, color="#808080", lw=0.8, ls="--")
    axes[1].set_xlabel("Business-count change")
    axes[1].set_ylabel("Diversity change")
    axes[1].set_title("Decline versus adaptation")
    axes[1].legend(loc="best")
    fig.suptitle("Grid-based typology of decline, relocation and adaptation", x=0.02, y=1.03, ha="left", fontsize=10, fontweight="bold")
    save_fig(fig, "fig_grid_250m_summary_typology")


def write_note():
    note = f"""# 250 m grid-based spatial analysis

Spatial unit: {GRID_SIZE} m x {GRID_SIZE} m grid cells in British National Grid (EPSG:27700).

Input records: retail records for Birmingham and Liverpool in 2016 and 2025.

Per-cell metrics:

- `business_count`: number of retail records in the grid cell.
- `vacancy_rate`: vacant / (occupied + vacant), excluding unknown occupation status from the denominator.
- `business_diversity`: Shannon diversity index across retail subgroups.
- `rateable_value_total`: sum of rateable value within the cell.
- `rateable_value_median`: median rateable value within the cell.

Change metrics compare 2025 against 2016. Cell status is defined from business-count change:

- `emerged`: no 2016 retail records, positive 2025 records.
- `disappeared`: positive 2016 records, no 2025 records.
- `intensified`: positive in both years, count increased.
- `declined`: positive in both years, count decreased.
- `persistent`: positive in both years, count unchanged.

Interpretive use:

- Decline is indicated by declining/disappeared cells, rising vacancy rate, and falling rateable value.
- Relocation is indicated by simultaneous disappeared/declined cells in one area and emerged/intensified cells elsewhere.
- Adaptation is indicated by stable or rising activity with increased business diversity or changed value structure.
"""
    (OUT / "README_grid_250m_spatial_analysis.md").write_text(note, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    points = load_retail_points()
    boundaries = load_boundaries_bng()
    grid, wide = aggregate_grid(points)
    grid.to_csv(OUT / "grid_250m_metrics_long_2016_2025.csv", index=False)
    wide.to_csv(OUT / "grid_250m_change_metrics_wide.csv", index=False)
    figure_grid_change(wide, boundaries)
    figure_grid_status(wide, boundaries)
    figure_grid_summary(wide)
    write_note()


if __name__ == "__main__":
    main()


