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
from pyproj import Transformer


INPUT = ROOT / "data" / "raw" / "openlocal_retail_property.parquet"
LAD_SHP = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.shp"
OUT = ROOT / "outputs" / "remade_progress_figures"

COLS = [
    "period",
    "geocode_name",
    "uarn",
    "occupation_state",
    "category_group",
    "category_subgroup",
    "primary_description",
    "geometry",
    "rateable_value",
    "total_floor_area",
]

CITY_ORDER = ["Birmingham", "Liverpool"]
CITY_COLORS = {"Birmingham": "#4C78A8", "Liverpool": "#F58518"}
STATE_COLORS = {"OCCUPIED": "#6F8F72", "VACANT": "#A95050", "UNKNOWN": "#BDBDBD"}
SUBGROUPS = [
    "RETAIL_HIGH_STREET",
    "RETAIL_RESTAURANTS_AND_CAFES",
    "RETAIL_OTHER_SUPERSTORES_AND_WAREHOUSES",
    "RETAIL_FINANCIAL_AND_PROFESSIONAL_SERVICES",
    "RETAIL_FOOD_SUPERSTORES",
]
SUBGROUP_LABELS = {
    "RETAIL_HIGH_STREET": "High-street retail",
    "RETAIL_RESTAURANTS_AND_CAFES": "Restaurants/cafes",
    "RETAIL_OTHER_SUPERSTORES_AND_WAREHOUSES": "Superstores/warehouses",
    "RETAIL_FINANCIAL_AND_PROFESSIONAL_SERVICES": "Financial/prof. services",
    "RETAIL_FOOD_SUPERSTORES": "Food superstores",
}
SUBGROUP_COLORS = {
    "RETAIL_HIGH_STREET": "#4C78A8",
    "RETAIL_RESTAURANTS_AND_CAFES": "#F58518",
    "RETAIL_OTHER_SUPERSTORES_AND_WAREHOUSES": "#6F8F72",
    "RETAIL_FINANCIAL_AND_PROFESSIONAL_SERVICES": "#D9A441",
    "RETAIL_FOOD_SUPERSTORES": "#A95050",
}

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


def load_retail():
    df = pq.read_table(INPUT, columns=COLS).to_pandas()
    df["period"] = pd.to_datetime(df["period"])
    df["year"] = df["period"].dt.year
    df = df[df["year"].between(2016, 2025)].copy()
    df = df[df["geocode_name"].isin(CITY_ORDER)].copy()
    df = df[df["category_group"].eq("RETAIL")].copy()
    df["occupation_clean"] = df["occupation_state"].fillna("UNKNOWN")
    df.loc[~df["occupation_clean"].isin(["OCCUPIED", "VACANT"]), "occupation_clean"] = "UNKNOWN"
    xy = np.array([parse_wkb_point(g) for g in df["geometry"]], dtype=float)
    df["lon"] = xy[:, 0]
    df["lat"] = xy[:, 1]
    df = df[np.isfinite(df["lon"]) & np.isfinite(df["lat"])].copy()
    df["floor_area_valid"] = df["total_floor_area"].where(df["total_floor_area"] > 0)
    return df


def load_boundaries():
    reader = shapefile.Reader(str(LAD_SHP))
    fields = [f.name for f in reader.fields[1:]]
    transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    out = {}
    for sr in reader.iterShapeRecords():
        rec = dict(zip(fields, sr.record))
        city = rec.get("LAD23NM")
        if city not in CITY_ORDER:
            continue
        pts = np.array(sr.shape.points, dtype=float)
        lon, lat = transformer.transform(pts[:, 0], pts[:, 1])
        parts = list(sr.shape.parts) + [len(pts)]
        out[city] = [np.column_stack([lon[s:e], lat[s:e]]) for s, e in zip(parts[:-1], parts[1:])]
    return out


def draw_boundary(ax, boundaries, city):
    for seg in boundaries.get(city, []):
        ax.plot(seg[:, 0], seg[:, 1], color="#2f2f2f", lw=0.85, alpha=0.9)


def save_fig(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ["svg", "pdf"]:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def line_panel(ax, data, y, ylabel, title, scale=1.0):
    for city in CITY_ORDER:
        g = data[data["geocode_name"].eq(city)]
        ax.plot(g["year"], g[y] / scale, marker="o", ms=3.5, lw=1.7, color=CITY_COLORS[city], label=city)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(2016, 2026, 2))
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)


def figure_scale_floor(df):
    stats = (
        df.groupby(["geocode_name", "year"], observed=True)
        .agg(
            retail_units=("uarn", "count"),
            valid_floor_records=("floor_area_valid", "count"),
            total_floor_area=("floor_area_valid", lambda s: s.sum(min_count=1)),
            median_floor_area=("floor_area_valid", "median"),
            q25_floor_area=("floor_area_valid", lambda s: s.quantile(0.25)),
            q75_floor_area=("floor_area_valid", lambda s: s.quantile(0.75)),
        )
        .reset_index()
    )
    stats["floor_area_coverage_pct"] = stats["valid_floor_records"] / stats["retail_units"] * 100
    stats.to_csv(OUT / "remade_retail_scale_floor_area_stats.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), constrained_layout=True)
    line_panel(axes[0, 0], stats, "retail_units", "Retail records", "Retail unit records")
    line_panel(axes[0, 1], stats, "total_floor_area", "Total floor area (million m虏)", "Aggregate floor area", scale=1_000_000)
    axes[0, 1].text(
        2017, 0.08, "2017 floor-area\nmissing", ha="center", va="bottom",
        fontsize=6.5, color="#666666",
    )
    for city in CITY_ORDER:
        g = stats[stats["geocode_name"].eq(city)]
        axes[1, 0].plot(g["year"], g["median_floor_area"], marker="o", ms=3.5, lw=1.7, color=CITY_COLORS[city], label=city)
        axes[1, 0].fill_between(g["year"], g["q25_floor_area"], g["q75_floor_area"], color=CITY_COLORS[city], alpha=0.16, lw=0)
        axes[1, 1].plot(g["year"], g["floor_area_coverage_pct"], marker="o", ms=3.5, lw=1.7, color=CITY_COLORS[city], label=city)
    axes[1, 0].set_title("Median floor area with IQR")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].set_ylabel("Floor area (m虏)")
    axes[1, 0].set_xticks(range(2016, 2026, 2))
    axes[1, 0].grid(axis="y", color="#E6E6E6", lw=0.6)
    axes[1, 1].axhline(100, color="#808080", lw=0.8, ls="--")
    axes[1, 1].set_title("Floor-area data coverage")
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Records with valid floor area (%)")
    axes[1, 1].set_xticks(range(2016, 2026, 2))
    axes[1, 1].set_ylim(-3, 103)
    axes[1, 1].grid(axis="y", color="#E6E6E6", lw=0.6)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Retail scale and floor-area change, 2016-2025", x=0.02, y=1.03, ha="left", fontsize=10, fontweight="bold")
    save_fig(fig, "fig01_retail_scale_floor_area")


def figure_occupation_quality(df):
    occ = df.groupby(["geocode_name", "year", "occupation_clean"], observed=True).size().rename("n").reset_index()
    occ.to_csv(OUT / "remade_occupation_status_counts.csv", index=False)
    totals = occ.groupby(["geocode_name", "year"], observed=True)["n"].transform("sum")
    occ["share"] = occ["n"] / totals * 100
    known = df[df["occupation_clean"].isin(["OCCUPIED", "VACANT"])]
    vacancy = (
        known.groupby(["geocode_name", "year"], observed=True)["occupation_clean"]
        .apply(lambda s: (s == "VACANT").mean() * 100)
        .rename("vacancy_rate_known")
        .reset_index()
    )
    unknown = occ[occ["occupation_clean"].eq("UNKNOWN")][["geocode_name", "year", "share"]].rename(columns={"share": "unknown_share"})
    vacancy.merge(unknown, on=["geocode_name", "year"], how="left").to_csv(OUT / "remade_vacancy_unknown_rates.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.1), constrained_layout=True)
    for ax, city in zip(axes[:2], CITY_ORDER):
        pivot = (
            occ[occ["geocode_name"].eq(city)]
            .pivot_table(index="year", columns="occupation_clean", values="share", aggfunc="sum")
            .reindex(range(2016, 2026))
            .fillna(0)
        )
        bottom = np.zeros(len(pivot.index))
        for state in ["OCCUPIED", "VACANT", "UNKNOWN"]:
            vals = pivot[state].to_numpy() if state in pivot else np.zeros(len(pivot.index))
            ax.bar(pivot.index, vals, bottom=bottom, color=STATE_COLORS[state], label=state.title())
            bottom += vals
        ax.set_title(city)
        ax.set_xlabel("Year")
        ax.set_ylabel("Share of retail records (%)")
        ax.set_xticks(range(2016, 2026, 3))
        ax.set_ylim(0, 100)
    for city in CITY_ORDER:
        g = vacancy[vacancy["geocode_name"].eq(city)]
        axes[2].plot(g["year"], g["vacancy_rate_known"], marker="o", ms=3.5, lw=1.7, color=CITY_COLORS[city], label=city)
    axes[2].set_title("Vacancy rate among known cases")
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("Vacant / known occupied-status records (%)")
    axes[2].set_xticks(range(2016, 2026, 2))
    axes[2].grid(axis="y", color="#E6E6E6", lw=0.6)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Status")
    axes[2].legend(loc="best")
    fig.suptitle("Occupation status and data-coverage sensitivity", x=0.02, y=1.04, ha="left", fontsize=10, fontweight="bold")
    save_fig(fig, "fig02_occupation_vacancy_data_quality")


def shannon_index(counts):
    vals = counts[counts > 0].to_numpy(dtype=float)
    p = vals / vals.sum()
    return float(-(p * np.log(p)).sum())


def figure_mix_shannon(df):
    mix = df[df["category_subgroup"].isin(SUBGROUPS)].groupby(["geocode_name", "year", "category_subgroup"], observed=True).size().rename("n").reset_index()
    totals = mix.groupby(["geocode_name", "year"], observed=True)["n"].transform("sum")
    mix["share"] = mix["n"] / totals * 100
    shannon = (
        mix.groupby(["geocode_name", "year"], observed=True)
        .apply(lambda g: shannon_index(g.set_index("category_subgroup")["n"]), include_groups=False)
        .rename("shannon_h")
        .reset_index()
    )
    mix.to_csv(OUT / "remade_retail_subgroup_mix.csv", index=False)
    shannon.to_csv(OUT / "remade_retail_subgroup_shannon.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True, height_ratios=[1.15, 1])
    for ax, city in zip(axes[0], CITY_ORDER):
        pivot = (
            mix[mix["geocode_name"].eq(city)]
            .pivot_table(index="year", columns="category_subgroup", values="share", aggfunc="sum")
            .reindex(range(2016, 2026))
            .fillna(0)
        )
        bottom = np.zeros(len(pivot.index))
        for subgroup in SUBGROUPS:
            vals = pivot[subgroup].to_numpy() if subgroup in pivot else np.zeros(len(pivot.index))
            ax.bar(pivot.index, vals, bottom=bottom, color=SUBGROUP_COLORS[subgroup], label=SUBGROUP_LABELS[subgroup])
            bottom += vals
        ax.set_title(city)
        ax.set_ylabel("Share of top retail subgroups (%)")
        ax.set_xlabel("Year")
        ax.set_ylim(0, 100)
        ax.set_xticks(range(2016, 2026, 3))
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Subgroup")
    for city in CITY_ORDER:
        g = shannon[shannon["geocode_name"].eq(city)]
        axes[1, 0].plot(g["year"], g["shannon_h"], marker="o", ms=3.5, lw=1.7, color=CITY_COLORS[city], label=city)
    axes[1, 0].set_title("Subgroup diversity")
    axes[1, 0].set_xlabel("Year")
    axes[1, 0].set_ylabel("Shannon diversity index")
    axes[1, 0].set_xticks(range(2016, 2026, 2))
    axes[1, 0].grid(axis="y", color="#E6E6E6", lw=0.6)
    axes[1, 0].legend(loc="best")
    change = mix[mix["year"].isin([2016, 2025])].pivot_table(index=["geocode_name", "category_subgroup"], columns="year", values="share", aggfunc="sum").fillna(0)
    change["change_pp"] = change[2025] - change[2016]
    change = change.reset_index()
    labels = [SUBGROUP_LABELS[s] for s in SUBGROUPS]
    x = np.arange(len(SUBGROUPS))
    width = 0.36
    for i, city in enumerate(CITY_ORDER):
        vals = change[change["geocode_name"].eq(city)].set_index("category_subgroup").reindex(SUBGROUPS)["change_pp"].fillna(0)
        axes[1, 1].bar(x + (i - 0.5) * width, vals, width=width, color=CITY_COLORS[city], label=city)
    axes[1, 1].axhline(0, color="#808080", lw=0.8)
    axes[1, 1].set_title("Change in subgroup share, 2016-2025")
    axes[1, 1].set_ylabel("Percentage-point change")
    axes[1, 1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1, 1].legend(loc="best")
    fig.suptitle("Retail adaptation: activity mix and diversity", x=0.02, y=1.03, ha="left", fontsize=10, fontweight="bold")
    save_fig(fig, "fig03_retail_mix_and_shannon")


def figure_floor_distribution(df):
    subset = df[df["year"].isin([2016, 2025])].copy()
    subset = subset[np.isfinite(subset["floor_area_valid"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True, constrained_layout=True)
    for ax, city in zip(axes, CITY_ORDER):
        arrays = []
        labels = []
        for year in [2016, 2025]:
            vals = subset[(subset["geocode_name"].eq(city)) & (subset["year"].eq(year))]["floor_area_valid"].dropna()
            vals = vals.clip(upper=vals.quantile(0.99))
            arrays.append(np.log10(vals.to_numpy()))
            labels.append(str(year))
        vp = ax.violinplot(arrays, positions=[1, 2], widths=0.7, showmedians=True, showextrema=False)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(CITY_COLORS[city])
            body.set_alpha(0.35 + 0.15 * i)
            body.set_edgecolor("none")
        vp["cmedians"].set_color("#222222")
        ax.set_xticks([1, 2], labels)
        ax.set_title(city)
        ax.set_xlabel("Year")
        ax.set_ylabel("log10 floor area (m虏)")
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    fig.suptitle("Distribution of retail floor area, 2016 versus 2025", x=0.02, y=1.03, ha="left", fontsize=10, fontweight="bold")
    save_fig(fig, "fig04_floor_area_distribution_2016_2025")


def figure_spatial_floor_change(df, boundaries):
    hs = df[df["category_subgroup"].eq("RETAIL_HIGH_STREET") & df["year"].isin([2016, 2025])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), constrained_layout=True)
    for ax, city in zip(axes, CITY_ORDER):
        g = hs[hs["geocode_name"].eq(city)].copy()
        xmin, xmax = g["lon"].quantile([0.005, 0.995])
        ymin, ymax = g["lat"].quantile([0.005, 0.995])
        nx = ny = 42
        xbins = np.linspace(xmin, xmax, nx + 1)
        ybins = np.linspace(ymin, ymax, ny + 1)
        grids = {}
        for year in [2016, 2025]:
            gy = g[g["year"].eq(year)]
            xi = np.digitize(gy["lon"], xbins) - 1
            yi = np.digitize(gy["lat"], ybins) - 1
            valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
            grid = np.zeros((ny, nx), dtype=float)
            for xx, yy, val in zip(xi[valid], yi[valid], gy.loc[valid, "floor_area_valid"].fillna(0)):
                grid[yy, xx] += val
            grids[year] = grid
        diff = grids[2025] - grids[2016]
        diff[np.isclose(grids[2025] + grids[2016], 0)] = np.nan
        vmax = np.nanpercentile(np.abs(diff), 97)
        im = ax.imshow(
            diff,
            extent=(xmin, xmax, ymin, ymax),
            origin="lower",
            cmap="RdBu",
            vmin=-vmax,
            vmax=vmax,
            alpha=0.82,
            interpolation="nearest",
        )
        draw_boundary(ax, boundaries, city)
        ax.scatter(g["lon"], g["lat"], s=0.35, c="#1f1f1f", alpha=0.08, rasterized=True, linewidths=0)
        ax.set_title(city)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="box")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Change in high-street retail floor area (m虏)")
    fig.suptitle("Spatial change in high-street retail floor area, 2016-2025", x=0.02, y=1.03, ha="left", fontsize=10, fontweight="bold")
    save_fig(fig, "fig05_spatial_floor_area_change")


def write_note():
    note = """# Remade dissertation progress figures

These figures replace the earlier standalone drafts with a consistent Python/matplotlib style and a clearer evidential structure.

- `fig01_retail_scale_floor_area`: combines unit records, total floor area, median/IQR floor area and valid floor-area data coverage.
- `fig02_occupation_vacancy_data_quality`: separates occupation-status composition from vacancy rate among known cases.
- `fig03_retail_mix_and_shannon`: combines retail-subgroup composition, Shannon diversity and 2016-2025 subgroup share change.
- `fig04_floor_area_distribution_2016_2025`: shows the floor-area distribution shift using log-scaled violins.
- `fig05_spatial_floor_area_change`: maps high-street retail floor-area change with LAD boundaries.

Important interpretation notes:

- Floor-area values are missing for 2017 in the provided source, so floor-area totals and medians should be interpreted from years with valid floor-area coverage.
- Unknown occupation status varies substantially over time, so vacancy-rate claims should use the known-status denominator and mention the changing unknown-status share as a data-coverage limitation.
"""
    (OUT / "README_remade_figures.md").write_text(note, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_retail()
    boundaries = load_boundaries()
    figure_scale_floor(df)
    figure_occupation_quality(df)
    figure_mix_shannon(df)
    figure_floor_distribution(df)
    figure_spatial_floor_change(df, boundaries)
    write_note()


if __name__ == "__main__":
    main()


