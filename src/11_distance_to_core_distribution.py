from pathlib import Path
import math
import struct

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
GRID_OUT = OUT / "grid_250m_spatial_analysis"

YEARS = [2016, 2019, 2022, 2025]
CITIES = ["Birmingham", "Liverpool"]

SCALE = 2
W, H = 1660, 780

PAL = {
    "bg": "#FFFFFF",
    "ink": "#25313A",
    "muted": "#66747C",
    "grid": "#E5ECEF",
    "axis": "#AAB6BD",
    "note": "#6C7880",
}

YEAR_COLORS = {
    2016: "#4C78A8",
    2019: "#4EA6A8",
    2022: "#F2A65A",
    2025: "#D46A6A",
}


def font(size, bold=False):
    paths = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size * SCALE)
    return ImageFont.load_default()


def text(draw, xy, s, size=20, fill=None, bold=False, anchor="la"):
    draw.text(
        (xy[0] * SCALE, xy[1] * SCALE),
        s,
        font=font(size, bold),
        fill=fill or PAL["ink"],
        anchor=anchor,
    )


def line(draw, pts, fill, width=2):
    draw.line([(x * SCALE, y * SCALE) for x, y in pts], fill=fill, width=width * SCALE, joint="curve")


def rect(draw, xy, fill, outline=None, width=1):
    draw.rectangle([v * SCALE for v in xy], fill=fill, outline=outline, width=width * SCALE)


def save_all(img, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / stem
    rgb = img.convert("RGB")
    rgb.save(base.with_suffix(".png"), dpi=(300, 300))
    rgb.save(base.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")
    rgb.save(base.with_suffix(".pdf"), "PDF", resolution=300)


def parse_wkb_point(hex_wkb):
    if not isinstance(hex_wkb, str) or len(hex_wkb) < 42:
        return math.nan, math.nan
    raw = bytes.fromhex(hex_wkb)
    endian = "<" if raw[0] == 1 else ">"
    geom_type = struct.unpack(endian + "I", raw[1:5])[0]
    if geom_type != 1:
        return math.nan, math.nan
    return struct.unpack(endian + "dd", raw[5:21])


def relative_xy(lon, lat, lon0, lat0):
    x = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110_540.0
    return x, y


def smooth(values):
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def load_points_with_distance():
    cores = pd.read_csv(OUT / "distance_to_core_reference_points.csv")
    core_lookup = {r.city: (float(r.core_lon), float(r.core_lat)) for r in cores.itertuples()}
    frames = []
    for city in CITIES:
        df = pd.read_csv(GRID_OUT / f"{city.lower()}_retail_points_selected_years_geometry.csv")
        df = df[df["year"].isin(YEARS)].copy()
        xy = [parse_wkb_point(g) for g in df["geometry"]]
        df["lon"] = [p[0] for p in xy]
        df["lat"] = [p[1] for p in xy]
        df = df.dropna(subset=["lon", "lat"]).copy()
        lon0, lat0 = core_lookup[city]
        coords = [relative_xy(lon, lat, lon0, lat0) for lon, lat in zip(df["lon"], df["lat"])]
        df["distance_km"] = [math.hypot(x, y) / 1000 for x, y in coords]
        df["city"] = city
        frames.append(df[["city", "year", "distance_km"]])
    return pd.concat(frames, ignore_index=True)


def distribution_table(df):
    bins = np.arange(0, 13.0, 0.5)
    centers = (bins[:-1] + bins[1:]) / 2
    rows = []
    for city in CITIES:
        for year in YEARS:
            vals = df[(df["city"].eq(city)) & (df["year"].eq(year))]["distance_km"].to_numpy()
            hist, _ = np.histogram(vals, bins=bins)
            share = hist / hist.sum() * 100 if hist.sum() else hist.astype(float)
            share_smooth = smooth(share)
            for center, raw, sm in zip(centers, share, share_smooth):
                rows.append({
                    "city": city,
                    "year": year,
                    "distance_km_bin_midpoint": center,
                    "share_percent": raw,
                    "smoothed_share_percent": sm,
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "distance_to_core_distribution_selected_years.csv", index=False)
    summary = df.groupby(["city", "year"])["distance_km"].agg(
        count="size",
        mean="mean",
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    ).reset_index()
    summary.to_csv(OUT / "distance_to_core_summary_selected_years.csv", index=False)
    return out, summary


def draw_panel(draw, dist, summary, city, box):
    x0, y0, w, h = box
    x_min, x_max = 0, 12
    y_min, y_max = 0, 16

    def tx(x):
        return x0 + (x - x_min) / (x_max - x_min) * w

    def ty(y):
        return y0 + h - (y - y_min) / (y_max - y_min) * h

    rect(draw, (x0, y0, x0 + w, y0 + h), "#FFFFFF")
    for yy in [0, 4, 8, 12, 16]:
        y = ty(yy)
        line(draw, [(x0, y), (x0 + w, y)], PAL["grid"], width=1)
        text(draw, (x0 - 14, y), str(yy), size=17, fill=PAL["muted"], anchor="rm")
    for xx in [0, 2, 4, 6, 8, 10, 12]:
        x = tx(xx)
        line(draw, [(x, y0), (x, y0 + h)], "#F1F5F7", width=1)
        text(draw, (x, y0 + h + 24), str(xx), size=17, fill=PAL["muted"], anchor="mm")

    line(draw, [(x0, y0 + h), (x0 + w, y0 + h)], PAL["axis"], width=2)
    line(draw, [(x0, y0), (x0, y0 + h)], PAL["axis"], width=2)
    text(draw, (x0 + w / 2, y0 - 36), city, size=30, bold=True, anchor="mm")
    text(draw, (x0, y0 - 16), "Share of retail records (%)", size=17, fill=PAL["ink"], anchor="la")
    text(draw, (x0 + w / 2, y0 + h + 78), "Distance to retail core (km)", size=19, fill=PAL["ink"], anchor="mm")

    for year in YEARS:
        sub = dist[(dist["city"].eq(city)) & (dist["year"].eq(year))]
        pts = [(tx(r.distance_km_bin_midpoint), ty(r.smoothed_share_percent)) for r in sub.itertuples()]
        line(draw, pts, YEAR_COLORS[year], width=5)
        med = float(summary[(summary["city"].eq(city)) & (summary["year"].eq(year))]["median"].iloc[0])
        mx = tx(med)
        line(draw, [(mx, y0 + h + 18), (mx, y0 + h + 38)], YEAR_COLORS[year], width=5)

    med_2016 = float(summary[(summary["city"].eq(city)) & (summary["year"].eq(2016))]["median"].iloc[0])
    med_2025 = float(summary[(summary["city"].eq(city)) & (summary["year"].eq(2025))]["median"].iloc[0])
    delta = med_2025 - med_2016
    label = f"Median shift: {delta:+.2f} km"
    text(draw, (x0 + w - 10, y0 + 20), label, size=18, fill=PAL["muted"], anchor="ra")


def draw_legend(draw, x, y):
    text(draw, (x, y), "Year", size=22, bold=True)
    for i, year in enumerate(YEARS):
        yy = y + 42 + i * 38
        line(draw, [(x, yy), (x + 54, yy)], YEAR_COLORS[year], width=5)
        text(draw, (x + 68, yy - 13), str(year), size=19)
    yy = y + 205
    line(draw, [(x, yy), (x + 54, yy)], PAL["muted"], width=4)
    text(draw, (x + 68, yy - 13), "Median tick", size=19, fill=PAL["muted"])


def main():
    df = load_points_with_distance()
    dist, summary = distribution_table(df)

    img = Image.new("RGBA", (W * SCALE, H * SCALE), PAL["bg"])
    draw = ImageDraw.Draw(img)

    draw_panel(draw, dist, summary, "Birmingham", (100, 95, 610, 570))
    draw_panel(draw, dist, summary, "Liverpool", (800, 95, 610, 570))
    draw_legend(draw, 1425, 300)
    save_all(img, "figure_distance_to_core_distribution_2016_2025")


if __name__ == "__main__":
    main()

