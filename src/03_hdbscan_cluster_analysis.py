import argparse
import math
import struct
from pathlib import Path

import hdbscan
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import linear_sum_assignment


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
    "postcode_id",
]


def parse_wkb_point(hex_wkb):
    if not isinstance(hex_wkb, str) or len(hex_wkb) < 42:
        return np.nan, np.nan
    raw = bytes.fromhex(hex_wkb)
    endian = "<" if raw[0] == 1 else ">"
    geom_type = struct.unpack(endian + "I", raw[1:5])[0]
    if geom_type != 1:
        return np.nan, np.nan
    return struct.unpack(endian + "dd", raw[5:21])


def load_retail(path):
    df = pq.read_table(path, columns=COLS).to_pandas()
    df["period"] = pd.to_datetime(df["period"])
    df["year"] = df["period"].dt.year
    df = df[df["geocode_name"].isin(["Birmingham", "Liverpool"])].copy()
    df = df[df["category_group"].eq("RETAIL")].copy()
    xy = np.array([parse_wkb_point(g) for g in df["geometry"]], dtype=float)
    df["lon"] = xy[:, 0]
    df["lat"] = xy[:, 1]
    df = df[np.isfinite(df["lon"]) & np.isfinite(df["lat"])].copy()
    # Local equirectangular metre coordinates are sufficient for intra-city
    # clustering over Birmingham/Liverpool extents and avoid a GIS dependency.
    for city, idx in df.groupby("geocode_name").groups.items():
        lon0 = df.loc[idx, "lon"].median()
        lat0 = df.loc[idx, "lat"].median()
        df.loc[idx, "x_m"] = (df.loc[idx, "lon"] - lon0) * 111_320.0 * math.cos(math.radians(lat0))
        df.loc[idx, "y_m"] = (df.loc[idx, "lat"] - lat0) * 110_540.0
    df["is_high_street"] = df["category_subgroup"].eq("RETAIL_HIGH_STREET")
    df["is_occupied"] = df["occupation_state"].eq("OCCUPIED")
    return df


def cluster_frame(part, min_cluster_size=None, min_samples=None):
    coords = part[["x_m", "y_m"]].to_numpy(dtype=float)
    if min_cluster_size is None:
        min_cluster_size = max(20, min(150, int(round(len(part) * 0.012))))
    if min_samples is None:
        min_samples = max(8, int(round(min_cluster_size * 0.35)))
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = model.fit_predict(coords)
    out = part.copy()
    out["cluster_id"] = labels
    out["cluster_probability"] = model.probabilities_
    out["min_cluster_size"] = min_cluster_size
    out["min_samples"] = min_samples
    return out


def summarize_clusters(assignments):
    clustered = assignments[assignments["cluster_id"] >= 0].copy()
    rows = []
    keys = ["city", "window", "cluster_id"]
    for key, g in clustered.groupby(keys, observed=True):
        city, window, cid = key
        subgroup_counts = g["category_subgroup"].value_counts(dropna=True)
        rows.append({
            "city": city,
            "window": window,
            "cluster_id": int(cid),
            "n_records": int(len(g)),
            "n_unique_uarn": int(g["uarn"].nunique()),
            "centroid_lon": float(g["lon"].mean()),
            "centroid_lat": float(g["lat"].mean()),
            "centroid_x_m": float(g["x_m"].mean()),
            "centroid_y_m": float(g["y_m"].mean()),
            "median_lon": float(g["lon"].median()),
            "median_lat": float(g["lat"].median()),
            "median_x_m": float(g["x_m"].median()),
            "median_y_m": float(g["y_m"].median()),
            "high_street_share": float(g["is_high_street"].mean()),
            "occupied_share": float(g["is_occupied"].mean()),
            "median_rateable_value": float(g["rateable_value"].median(skipna=True)),
            "median_floor_area": float(g["total_floor_area"].median(skipna=True)),
            "dominant_subgroup": subgroup_counts.index[0] if len(subgroup_counts) else "",
            "dominant_subgroup_share": float(subgroup_counts.iloc[0] / len(g)) if len(subgroup_counts) else np.nan,
        })
    return pd.DataFrame(rows)


def build_movements(summary, max_match_distance=1200.0):
    rows = []
    for city in sorted(summary["city"].unique()):
        early = summary[(summary.city == city) & (summary.window == "2016")].reset_index(drop=True)
        late = summary[(summary.city == city) & (summary.window == "2025")].reset_index(drop=True)
        matched_early = set()
        matched_late = set()
        if len(early) and len(late):
            a = early[["centroid_x_m", "centroid_y_m"]].to_numpy()
            b = late[["centroid_x_m", "centroid_y_m"]].to_numpy()
            dist = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
            ridx, cidx = linear_sum_assignment(dist)
            for i, j in zip(ridx, cidx):
                d = float(dist[i, j])
                if d <= max_match_distance:
                    matched_early.add(i)
                    matched_late.add(j)
                    e = early.loc[i]
                    l = late.loc[j]
                    dx = float(l.centroid_x_m - e.centroid_x_m)
                    dy = float(l.centroid_y_m - e.centroid_y_m)
                    rows.append({
                        "city": city,
                        "early_cluster_id": int(e.cluster_id),
                        "late_cluster_id": int(l.cluster_id),
                        "status": "persistent_or_relocated",
                        "distance_m": d,
                        "dx_m": dx,
                        "dy_m": dy,
                        "early_n_records": int(e.n_records),
                        "late_n_records": int(l.n_records),
                        "record_change_pct": float((l.n_records - e.n_records) / e.n_records * 100.0),
                        "early_high_street_share": float(e.high_street_share),
                        "late_high_street_share": float(l.high_street_share),
                        "early_centroid_lon": float(e.centroid_lon),
                        "early_centroid_lat": float(e.centroid_lat),
                        "late_centroid_lon": float(l.centroid_lon),
                        "late_centroid_lat": float(l.centroid_lat),
                        "early_centroid_x_m": float(e.centroid_x_m),
                        "early_centroid_y_m": float(e.centroid_y_m),
                        "late_centroid_x_m": float(l.centroid_x_m),
                        "late_centroid_y_m": float(l.centroid_y_m),
                    })
        for i, e in early.iterrows():
            if i not in matched_early:
                rows.append({
                    "city": city,
                    "early_cluster_id": int(e.cluster_id),
                    "late_cluster_id": np.nan,
                    "status": "disappeared",
                    "distance_m": np.nan,
                    "dx_m": np.nan,
                    "dy_m": np.nan,
                    "early_n_records": int(e.n_records),
                    "late_n_records": 0,
                    "record_change_pct": -100.0,
                    "early_high_street_share": float(e.high_street_share),
                    "late_high_street_share": np.nan,
                    "early_centroid_lon": float(e.centroid_lon),
                    "early_centroid_lat": float(e.centroid_lat),
                    "late_centroid_lon": np.nan,
                    "late_centroid_lat": np.nan,
                    "early_centroid_x_m": float(e.centroid_x_m),
                    "early_centroid_y_m": float(e.centroid_y_m),
                    "late_centroid_x_m": np.nan,
                    "late_centroid_y_m": np.nan,
                })
        for j, l in late.iterrows():
            if j not in matched_late:
                rows.append({
                    "city": city,
                    "early_cluster_id": np.nan,
                    "late_cluster_id": int(l.cluster_id),
                    "status": "emerged",
                    "distance_m": np.nan,
                    "dx_m": np.nan,
                    "dy_m": np.nan,
                    "early_n_records": 0,
                    "late_n_records": int(l.n_records),
                    "record_change_pct": np.nan,
                    "early_high_street_share": np.nan,
                    "late_high_street_share": float(l.high_street_share),
                    "early_centroid_lon": np.nan,
                    "early_centroid_lat": np.nan,
                    "late_centroid_lon": float(l.centroid_lon),
                    "late_centroid_lat": float(l.centroid_lat),
                    "early_centroid_x_m": np.nan,
                    "early_centroid_y_m": np.nan,
                    "late_centroid_x_m": float(l.centroid_x_m),
                    "late_centroid_y_m": float(l.centroid_y_m),
                })
    return pd.DataFrame(rows)


def add_north_equal(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")


def plot_results(assignments, summary, movements, output_prefix, filter_label):
    colors = {"2016": "#4C78A8", "2025": "#F58518"}
    fig = plt.figure(figsize=(7.2, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0])
    axes = {
        "Birmingham": fig.add_subplot(gs[0, 0]),
        "Liverpool": fig.add_subplot(gs[0, 1]),
        "bar": fig.add_subplot(gs[1, 0]),
        "move": fig.add_subplot(gs[1, 1]),
    }
    for city, ax in axes.items():
        if city not in ["Birmingham", "Liverpool"]:
            continue
        city_assign = assignments[assignments["city"].eq(city)]
        noise = city_assign[city_assign["cluster_id"].eq(-1)]
        ax.scatter(noise["lon"], noise["lat"], s=0.7, c="#D9D9D9", alpha=0.18, rasterized=True, linewidths=0)
        for window, marker in [("2016", "o"), ("2025", "^")]:
            part = summary[(summary.city == city) & (summary.window == window)]
            sizes = np.clip(np.sqrt(part["n_records"]) * 7.0, 20, 150)
            ax.scatter(
                part["centroid_lon"], part["centroid_lat"],
                s=sizes, c=colors[window], marker=marker, alpha=0.84,
                edgecolors="white", linewidths=0.45, label=window,
            )
        city_moves = movements[(movements.city == city) & (movements.status == "persistent_or_relocated")]
        for _, r in city_moves.iterrows():
            if r["distance_m"] >= 150:
                ax.annotate(
                    "",
                    xy=(r.late_centroid_lon, r.late_centroid_lat),
                    xytext=(r.early_centroid_lon, r.early_centroid_lat),
                    arrowprops=dict(arrowstyle="->", color="#4A4A4A", lw=0.7, alpha=0.65),
                )
        ax.set_title(city)
        add_north_equal(ax)
        ax.legend(loc="best", title="Cluster centroids")

    counts = movements.groupby(["city", "status"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["persistent_or_relocated", "emerged", "disappeared"], fill_value=0)
    x = np.arange(len(counts.index))
    bottom = np.zeros(len(counts.index))
    status_colors = {
        "persistent_or_relocated": "#6F8F72",
        "emerged": "#D9A441",
        "disappeared": "#A95050",
    }
    labels = {
        "persistent_or_relocated": "persistent / relocated",
        "emerged": "emerged",
        "disappeared": "disappeared",
    }
    for status in counts.columns:
        axes["bar"].bar(x, counts[status], bottom=bottom, color=status_colors[status], label=labels[status])
        bottom += counts[status].to_numpy()
    axes["bar"].set_xticks(x, counts.index)
    axes["bar"].set_ylabel("Number of HDBSCAN clusters")
    axes["bar"].set_title("Cluster status, 2016 to 2025")
    axes["bar"].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)

    moved = movements[movements["status"].eq("persistent_or_relocated")].copy()
    for city, color in [("Birmingham", "#4C78A8"), ("Liverpool", "#F58518")]:
        part = moved[moved.city.eq(city)]
        axes["move"].scatter(
            part["distance_m"], part["record_change_pct"],
            s=np.clip(np.sqrt(part["late_n_records"].clip(lower=1)) * 8, 25, 150),
            c=color, alpha=0.72, edgecolors="white", linewidths=0.5, label=city,
        )
    axes["move"].axhline(0, color="#808080", lw=0.8, ls="--")
    axes["move"].set_xlabel("Centroid shift among matched clusters (m)")
    axes["move"].set_ylabel("Cluster record change, 2016 to 2025 (%)")
    axes["move"].set_title("Relocation distance versus activity change")
    axes["move"].legend(loc="best")

    fig.suptitle(
        f"HDBSCAN identifies persistence, emergence and relocation of {filter_label} clusters",
        x=0.02, y=1.02, ha="left", fontsize=10, fontweight="bold",
    )
    fig.savefig(f"{output_prefix}.svg", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{output_prefix}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--filter", choices=["retail", "retail_high_street"], default="retail")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_retail(args.input)
    if args.filter == "retail_high_street":
        df = df[df["category_subgroup"].eq("RETAIL_HIGH_STREET")].copy()
    df = df[df["year"].isin([2016, 2025])].copy()
    df["window"] = df["year"].astype(str)
    df["city"] = df["geocode_name"]

    assignments = []
    parameters = []
    for (city, window), part in df.groupby(["city", "window"], observed=True):
        clustered = cluster_frame(part)
        assignments.append(clustered)
        parameters.append({
            "city": city,
            "window": window,
            "n_records": len(part),
            "min_cluster_size": int(clustered["min_cluster_size"].iloc[0]),
            "min_samples": int(clustered["min_samples"].iloc[0]),
            "noise_share": float((clustered["cluster_id"] == -1).mean()),
            "n_clusters": int(clustered.loc[clustered["cluster_id"] >= 0, "cluster_id"].nunique()),
        })
    assignments = pd.concat(assignments, ignore_index=True)
    summary = summarize_clusters(assignments)
    movements = build_movements(summary)
    parameters = pd.DataFrame(parameters)

    base = f"hdbscan_{args.filter}_clusters"
    keep = [
        "period", "city", "window", "uarn", "occupation_state", "category_subgroup",
        "primary_description", "postcode_id", "lon", "lat", "x_m", "y_m", "cluster_id",
        "cluster_probability", "rateable_value", "total_floor_area",
    ]
    assignments[keep].to_csv(outdir / f"{base}_assignments.csv", index=False)
    summary.to_csv(outdir / f"{base}_summary.csv", index=False)
    movements.to_csv(outdir / f"{base}_movements.csv", index=False)
    parameters.to_csv(outdir / f"{base}_parameters.csv", index=False)
    label = "retail-related" if args.filter == "retail" else "high-street retail"
    plot_results(assignments, summary, movements, outdir / base, label)

    print("records", len(assignments))
    print(parameters.to_string(index=False))
    print("outputs", outdir)


if __name__ == "__main__":
    main()

