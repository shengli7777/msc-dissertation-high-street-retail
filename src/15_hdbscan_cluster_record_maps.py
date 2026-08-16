from pathlib import Path
import colorsys
import math
import struct

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs"
OUT = ROOT / "work" / "cluster_figures_4_14_4_17"
OUT.mkdir(parents=True, exist_ok=True)
ASSIGN_CSV = DATA / "hdbscan_retail_clusters_assignments.csv"
LAD_SHP = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.shp"
LAD_DBF = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.dbf"

CITY_ORDER = ["Birmingham", "Liverpool"]
YEARS = [2016, 2025]
SCALE = 2
W, H = 2050, 4300
LIVERPOOL_HALF_PAGE_SCALE = None

PAL = {
    "bg": "#FFFFFF",
    "ink": "#1F2933",
    "muted": "#667085",
    "boundary": "#7A8791",
    "panel": "#FFFFFF",
    "panel_line": "#111111",
    "dark": "#111827",
    "point_bg": "#D9E3EA",
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


F_PANEL = font(36, bold=True)
F_SINGLE_TITLE = font(58, bold=True)
F_SINGLE_LEG = font(31)
F_SINGLE_LEG_BOLD = font(35, bold=True)
F_SINGLE_NOTE = font(27)
F_SMALL = font(22)
F_TINY = font(18)
F_LEG = font(21)
F_LEG_BOLD = font(23, bold=True)


def sc(v):
    return v * SCALE


def text(draw, xy, s, fnt, fill=PAL["ink"], anchor="la"):
    draw.text((sc(xy[0]), sc(xy[1])), s, font=fnt, fill=fill, anchor=anchor)


def line(draw, pts, fill, width=1):
    draw.line([(sc(x), sc(y)) for x, y in pts], fill=fill, width=max(1, round(width * SCALE)))


def circle(draw, x, y, r, fill, outline=None, width=1):
    draw.ellipse(
        (sc(x - r), sc(y - r), sc(x + r), sc(y + r)),
        fill=fill,
        outline=outline or fill,
        width=max(1, round(width * SCALE)),
    )


def save_all(img, stem):
    base = OUT / stem
    rgb = img.convert("RGB")
    rgb.save(base.with_suffix(".png"), dpi=(300, 300))
    rgb.save(base.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")
    rgb.save(base.with_suffix(".pdf"), "PDF", resolution=300)


def read_dbf_records(path):
    with open(path, "rb") as f:
        header = f.read(32)
        n_records = struct.unpack("<I", header[4:8])[0]
        header_len = struct.unpack("<H", header[8:10])[0]
        record_len = struct.unpack("<H", header[10:12])[0]
        fields = []
        while f.tell() < header_len - 1:
            desc = f.read(32)
            if not desc or desc[0] == 0x0D:
                break
            name = desc[:11].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
            length = desc[16]
            fields.append((name, length))
        f.seek(header_len)
        records = []
        for _ in range(n_records):
            raw = f.read(record_len)
            if not raw or raw[:1] == b"*":
                continue
            pos = 1
            rec = {}
            for name, length in fields:
                rec[name] = raw[pos:pos + length].decode("latin1", errors="ignore").strip()
                pos += length
            records.append(rec)
    return records


def read_shp_polygon_segments(path):
    segments_by_record = []
    with open(path, "rb") as f:
        f.seek(100)
        while True:
            rec_header = f.read(8)
            if len(rec_header) < 8:
                break
            content_len_words = struct.unpack(">i", rec_header[4:8])[0]
            content = f.read(content_len_words * 2)
            if len(content) < 44:
                segments_by_record.append([])
                continue
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type not in (5, 15, 25):
                segments_by_record.append([])
                continue
            num_parts, num_points = struct.unpack("<2i", content[36:44])
            parts_start = 44
            parts = list(struct.unpack("<" + "i" * num_parts, content[parts_start:parts_start + 4 * num_parts]))
            points_start = parts_start + 4 * num_parts
            points = []
            for i in range(num_points):
                x, y = struct.unpack("<2d", content[points_start + i * 16:points_start + (i + 1) * 16])
                points.append((x, y))
            parts.append(num_points)
            segments_by_record.append([points[s:e] for s, e in zip(parts[:-1], parts[1:])])
    return segments_by_record


def osgb36_to_lonlat(easting, northing):
    a = 6377563.396
    b = 6356256.909
    f0 = 0.9996012717
    lat0 = math.radians(49)
    lon0 = math.radians(-2)
    n0 = -100000
    e0 = 400000
    e2 = 1 - (b * b) / (a * a)
    n = (a - b) / (a + b)
    lat = lat0
    m = 0
    while abs(northing - n0 - m) >= 0.00001:
        lat = (northing - n0 - m) / (a * f0) + lat
        ma = (1 + n + 5 / 4 * n**2 + 5 / 4 * n**3) * (lat - lat0)
        mb = (3 * n + 3 * n**2 + 21 / 8 * n**3) * math.sin(lat - lat0) * math.cos(lat + lat0)
        mc = (15 / 8 * n**2 + 15 / 8 * n**3) * math.sin(2 * (lat - lat0)) * math.cos(2 * (lat + lat0))
        md = 35 / 24 * n**3 * math.sin(3 * (lat - lat0)) * math.cos(3 * (lat + lat0))
        m = b * f0 * (ma - mb + mc - md)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)
    nu = a * f0 / math.sqrt(1 - e2 * sin_lat**2)
    rho = a * f0 * (1 - e2) / (1 - e2 * sin_lat**2) ** 1.5
    eta2 = nu / rho - 1
    d_e = easting - e0
    vii = tan_lat / (2 * rho * nu)
    viii = tan_lat / (24 * rho * nu**3) * (5 + 3 * tan_lat**2 + eta2 - 9 * tan_lat**2 * eta2)
    ix = tan_lat / (720 * rho * nu**5) * (61 + 90 * tan_lat**2 + 45 * tan_lat**4)
    x = 1 / (cos_lat * nu)
    xi = 1 / (6 * cos_lat * nu**3) * (nu / rho + 2 * tan_lat**2)
    xii = 1 / (120 * cos_lat * nu**5) * (5 + 28 * tan_lat**2 + 24 * tan_lat**4)
    xiia = 1 / (5040 * cos_lat * nu**7) * (61 + 662 * tan_lat**2 + 1320 * tan_lat**4 + 720 * tan_lat**6)
    lat = lat - vii * d_e**2 + viii * d_e**4 - ix * d_e**6
    lon = lon0 + x * d_e - xi * d_e**3 + xii * d_e**5 - xiia * d_e**7
    return math.degrees(lon), math.degrees(lat)


def relative_xy(assign, city, lon, lat):
    city_points = assign[assign.city.eq(city)]
    lon0 = city_points["lon"].median()
    lat0 = city_points["lat"].median()
    x = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110_540.0
    return x, y


def load_lad_boundaries(assign):
    if not (LAD_SHP.exists() and LAD_DBF.exists()):
        return {}
    records = read_dbf_records(LAD_DBF)
    segments = read_shp_polygon_segments(LAD_SHP)
    out = {}
    for rec, segs in zip(records, segments):
        city = rec.get("LAD23NM")
        if city in CITY_ORDER:
            city_segments = []
            for seg in segs:
                converted = []
                for east, north in seg:
                    lon, lat = osgb36_to_lonlat(east, north)
                    converted.append(relative_xy(assign, city, lon, lat))
                city_segments.append(converted)
            out[city] = city_segments
    return out


def city_extent(assign, boundaries, city):
    pts = assign.loc[assign.city.eq(city), ["x_m", "y_m"]].dropna()
    xs = pts.x_m.tolist()
    ys = pts.y_m.tolist()
    for seg in boundaries.get(city, []):
        xs.extend([x for x, _ in seg])
        ys.extend([y for _, y in seg])
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_x = (xmax - xmin) * 0.035
    pad_y = (ymax - ymin) * 0.035
    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def make_transform(extent, rect, fixed_scale=None, align_x=0.5):
    xmin, xmax, ymin, ymax = extent
    x0, y0, w, h = rect
    s = fixed_scale if fixed_scale is not None else min(w / (xmax - xmin), h / (ymax - ymin))
    used_w = (xmax - xmin) * s
    used_h = (ymax - ymin) * s
    ox = x0 + (w - used_w) * align_x
    oy = y0 + (h - used_h) / 2

    def tr(x, y):
        return ox + (x - xmin) * s, oy + used_h - (y - ymin) * s

    return tr, s


def draw_boundary(draw, boundary, tr):
    for seg in boundary:
        pts = [tr(x, y) for x, y in seg]
        if len(pts) > 1:
            line(draw, pts, PAL["boundary"], width=1.5)


def draw_north(draw, x, y, size=28):
    pts = [(x, y - size), (x - size * 0.35, y + size * 0.45), (x, y + size * 0.12), (x + size * 0.35, y + size * 0.45)]
    draw.polygon([(sc(px), sc(py)) for px, py in pts], fill="white", outline=PAL["dark"])
    inner = [(x, y - size * 0.70), (x - size * 0.10, y + size * 0.05), (x, y - size * 0.10), (x + size * 0.10, y + size * 0.05)]
    draw.polygon([(sc(px), sc(py)) for px, py in inner], fill=PAL["dark"])
    text(draw, (x, y + size * 0.68), "N", F_TINY, PAL["dark"], anchor="mm")


def draw_scale(draw, rect, scale, length=5000):
    x0, y0, w, h = rect
    x1 = x0 + 72
    y1 = y0 + h - 54
    x2 = x1 + length * scale
    line(draw, [(x1, y1), (x2, y1)], PAL["dark"], width=3.6)
    for xx in [x1, (x1 + x2) / 2, x2]:
        line(draw, [(xx, y1 - 10), (xx, y1 + 10)], PAL["dark"], width=2.2)
    text(draw, (x1, y1 + 26), "0", F_TINY, PAL["dark"], anchor="ma")
    mid_label = f"{length / 2000:g}"
    end_label = f"{length / 1000:g} km"
    text(draw, ((x1 + x2) / 2, y1 + 26), mid_label, F_TINY, PAL["dark"], anchor="ma")
    text(draw, (x2, y1 + 26), end_label, F_TINY, PAL["dark"], anchor="ma")


def cluster_palette(n=40):
    colors = []
    for i in range(n):
        hue = (0.08 + i * 0.61803398875) % 1.0
        sat = 0.58 if i % 3 else 0.70
        val = 0.78 if i % 4 else 0.66
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append((int(r * 255), int(g * 255), int(b * 255), 178))
    return colors


def draw_clustered_points(layer, rows, tr, colors):
    draw = ImageDraw.Draw(layer, "RGBA")
    cluster_ids = sorted([int(v) for v in rows.cluster_id.unique() if int(v) != -1])
    color_map = {cid: colors[i % len(colors)] for i, cid in enumerate(cluster_ids)}
    rows = rows[rows.cluster_id.ne(-1)].copy()
    rows["_cluster_size"] = rows.groupby("cluster_id")["cluster_id"].transform("size")
    rows = rows.sort_values(["_cluster_size", "cluster_id"], ascending=[False, True])
    for row in rows.itertuples():
        px, py = tr(row.x_m, row.y_m)
        c = color_map[int(row.cluster_id)]
        r = 2.45
        draw.ellipse((sc(px - r), sc(py - r), sc(px + r), sc(py + r)), fill=c, outline=None)


def draw_panel_legend(draw, rect):
    x0, y0, w, h = rect
    x = x0 + w - 545
    y = y0 + h - 230
    text(draw, (x, y), "HDBSCAN clusters", F_LEG_BOLD, PAL["ink"])
    text(draw, (x, y + 42), "Colours distinguish clusters", F_LEG, PAL["ink"])
    text(draw, (x, y + 78), "within each panel only", F_LEG, PAL["ink"])
    line(draw, [(x, y + 124), (x + 86, y + 124)], PAL["boundary"], width=1.8)
    text(draw, (x + 106, y + 109), "Local authority boundary", F_LEG, PAL["ink"])
    text(draw, (x, y + 166), "Noise records not shown", F_TINY, PAL["muted"])


def draw_panel_legend_large(draw, rect):
    x0, y0, w, h = rect
    x = x0 + w - 610
    y = y0 + h - 560
    text(draw, (x, y), "HDBSCAN clusters", F_SINGLE_LEG_BOLD, PAL["ink"])
    text(draw, (x, y + 62), "Colours distinguish clusters", F_SINGLE_LEG, PAL["ink"])
    text(draw, (x, y + 110), "within each panel only", F_SINGLE_LEG, PAL["ink"])
    line(draw, [(x, y + 175), (x + 120, y + 175)], PAL["boundary"], width=2.3)
    text(draw, (x + 150, y + 151), "Local authority boundary", F_SINGLE_LEG, PAL["ink"])
    text(draw, (x, y + 250), "Noise records not shown", F_SINGLE_NOTE, PAL["muted"])


def main():
    assign = pd.read_csv(ASSIGN_CSV)
    boundaries = load_lad_boundaries(assign)
    panels = {
        ("Birmingham", 2016): (95, 120, 1860, 900),
        ("Birmingham", 2025): (95, 1155, 1860, 900),
        ("Liverpool", 2016): (95, 2190, 1860, 900),
        ("Liverpool", 2025): (95, 3225, 1860, 900),
    }
    extents = {city: city_extent(assign, boundaries, city) for city in CITY_ORDER}
    common_scale = None
    for city in CITY_ORDER:
        xmin, xmax, ymin, ymax = extents[city]
        for year in YEARS:
            _, _, w, h = panels[(city, year)]
            city_scale = min(w / (xmax - xmin), h / (ymax - ymin))
            common_scale = city_scale if common_scale is None else min(common_scale, city_scale)

    img = Image.new("RGBA", (W * SCALE, H * SCALE), PAL["bg"])
    draw = ImageDraw.Draw(img)
    colors = cluster_palette(48)

    for city in CITY_ORDER:
        for year in YEARS:
            rect = panels[(city, year)]
            draw.rectangle((sc(rect[0]), sc(rect[1]), sc(rect[0] + rect[2]), sc(rect[1] + rect[3])), fill=PAL["panel"], outline=PAL["panel_line"], width=3 * SCALE)
            text(draw, (rect[0], rect[1] - 54), f"{city}, {year}", F_PANEL, PAL["dark"])
            tr, scale = make_transform(extents[city], rect, fixed_scale=common_scale)
            panel_rows = assign[(assign.city.eq(city)) & (assign.window.eq(year))]
            point_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
            draw_clustered_points(point_layer, panel_rows, tr, colors)
            img.alpha_composite(point_layer)
            draw_boundary(draw, boundaries.get(city, []), tr)
            draw_north(draw, rect[0] + 78, rect[1] + 118, size=48)
            draw_scale(draw, rect, scale)
            draw_panel_legend(draw, rect)

    save_all(img, "figure_4_12_hdbscan_clustered_retail_records_distribution")

    single_panel = (75, 120, 2260, 1400)
    _, birmingham_single_scale = make_transform(extents["Birmingham"], single_panel, fixed_scale=None, align_x=0.31)
    for city in CITY_ORDER:
        for year in YEARS:
            single_img = Image.new("RGBA", (2500 * SCALE, 1700 * SCALE), PAL["bg"])
            single_draw = ImageDraw.Draw(single_img)
            single_draw.rectangle(
                (
                    sc(single_panel[0]),
                    sc(single_panel[1]),
                    sc(single_panel[0] + single_panel[2]),
                    sc(single_panel[1] + single_panel[3]),
                ),
                fill=PAL["panel"],
                outline=PAL["panel_line"],
                width=3 * SCALE,
            )
            text(
                single_draw,
                (single_panel[0] + single_panel[2] / 2, single_panel[1] - 62),
                f"{city}, {year}",
                F_SINGLE_TITLE,
                PAL["dark"],
                anchor="mm",
            )
            panel_rows = assign[(assign.city.eq(city)) & (assign.window.eq(year))]
            point_layer = Image.new("RGBA", single_img.size, (255, 255, 255, 0))
            if city == "Liverpool":
                tr, scale = make_transform(extents[city], single_panel, fixed_scale=birmingham_single_scale, align_x=0.31)
                draw_clustered_points(point_layer, panel_rows, tr, colors)
                single_img.alpha_composite(point_layer)
                draw_boundary(single_draw, boundaries.get(city, []), tr)
            else:
                tr, scale = make_transform(extents[city], single_panel, fixed_scale=None, align_x=0.31)
                draw_clustered_points(point_layer, panel_rows, tr, colors)
                single_img.alpha_composite(point_layer)
                draw_boundary(single_draw, boundaries.get(city, []), tr)
            draw_north(single_draw, single_panel[0] + 105, single_panel[1] + 150, size=66)
            draw_scale(single_draw, single_panel, scale, length=5000)
            draw_panel_legend_large(single_draw, single_panel)
            save_all(
                single_img,
                f"figure_4_12_hdbscan_clustered_records_{city.lower()}_{year}",
            )
    print("Wrote", OUT / "figure_4_12_hdbscan_clustered_retail_records_distribution.png")


if __name__ == "__main__":
    main()

