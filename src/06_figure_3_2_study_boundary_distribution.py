from pathlib import Path
import math
import struct
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs"
OUT = ROOT / "work" / "location_figures"
OUT.mkdir(parents=True, exist_ok=True)

LAD_SHP = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.shp"
LAD_DBF = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.dbf"
POINTS = DATA / "hdbscan_retail_clusters_assignments.csv"

CITY_ORDER = ["Birmingham", "Liverpool"]
PAL = {
    "bg": "#FFFFFF",
    "panel": "#FFFFFF",
    "ink": "#1F2933",
    "muted": "#667085",
    "boundary": "#9FAEB7",
    "point": "#0B6172",
    "point_edge": "#FFFFFF",
    "frame": "#111111",
}


def font(size, bold=False):
    paths = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_PANEL = font(36, True)
F_SMALL = font(22)
F_TINY = font(17)


def text(draw, xy, value, fnt, fill=PAL["ink"], anchor="la"):
    draw.text(xy, value, font=fnt, fill=fill, anchor=anchor)


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
            fields.append((name, desc[16]))
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
            header = f.read(8)
            if len(header) < 8:
                break
            content_len_words = struct.unpack(">i", header[4:8])[0]
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
            rec_segments = []
            for a, b in zip(parts[:-1], parts[1:]):
                seg = points[a:b]
                if len(seg) > 2:
                    rec_segments.append(seg)
            segments_by_record.append(rec_segments)
    return segments_by_record


def osgb36_to_lonlat(easting, northing):
    a = 6377563.396
    b = 6356256.909
    f0 = 0.9996012717
    lat0 = math.radians(49.0)
    lon0 = math.radians(-2.0)
    n0 = -100000.0
    e0 = 400000.0
    e2 = 1 - (b * b) / (a * a)
    n = (a - b) / (a + b)
    lat = lat0
    m = 0
    while abs(northing - n0 - m) >= 0.00001:
        lat = (northing - n0 - m) / (a * f0) + lat
        ma = (1 + n + 1.25 * n**2 + 1.25 * n**3) * (lat - lat0)
        mb = (3 * n + 3 * n**2 + 2.625 * n**3) * math.sin(lat - lat0) * math.cos(lat + lat0)
        mc = (1.875 * n**2 + 1.875 * n**3) * math.sin(2 * (lat - lat0)) * math.cos(2 * (lat + lat0))
        md = (35 / 24 * n**3) * math.sin(3 * (lat - lat0)) * math.cos(3 * (lat + lat0))
        m = b * f0 * (ma - mb + mc - md)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    nu = a * f0 / math.sqrt(1 - e2 * sin_lat**2)
    rho = a * f0 * (1 - e2) / (1 - e2 * sin_lat**2) ** 1.5
    eta2 = nu / rho - 1
    tan_lat = math.tan(lat)
    sec_lat = 1 / cos_lat
    d_e = easting - e0
    vii = tan_lat / (2 * rho * nu)
    viii = tan_lat / (24 * rho * nu**3) * (5 + 3 * tan_lat**2 + eta2 - 9 * tan_lat**2 * eta2)
    ix = tan_lat / (720 * rho * nu**5) * (61 + 90 * tan_lat**2 + 45 * tan_lat**4)
    x = sec_lat / nu
    xi = sec_lat / (6 * nu**3) * (nu / rho + 2 * tan_lat**2)
    xii = sec_lat / (120 * nu**5) * (5 + 28 * tan_lat**2 + 24 * tan_lat**4)
    xii_a = sec_lat / (5040 * nu**7) * (61 + 662 * tan_lat**2 + 1320 * tan_lat**4 + 720 * tan_lat**6)
    lat = lat - vii * d_e**2 + viii * d_e**4 - ix * d_e**6
    lon = lon0 + x * d_e - xi * d_e**3 + xii * d_e**5 - xii_a * d_e**7
    return math.degrees(lon), math.degrees(lat)


def relative_xy(city, lon, lat, centres):
    lon0, lat0 = centres[city]
    x = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110_540.0
    return x, y


def load_city_boundaries(centres):
    records = read_dbf_records(LAD_DBF)
    segments = read_shp_polygon_segments(LAD_SHP)
    out = {}
    for rec, segs in zip(records, segments):
        city = rec.get("LAD23NM")
        if city in CITY_ORDER:
            converted = []
            for seg in segs:
                converted.append([relative_xy(city, *osgb36_to_lonlat(e, n), centres) for e, n in seg])
            out[city] = converted
    return out


def simplify_segment(seg, step=4):
    if len(seg) <= step:
        return seg
    out = seg[::step]
    if out[-1] != seg[-1]:
        out.append(seg[-1])
    return out


def extent_for(city, points, boundaries):
    city_pts = points.loc[points.city.eq(city), ["x_m", "y_m"]].dropna()
    xs = city_pts.x_m.tolist()
    ys = city_pts.y_m.tolist()
    for seg in boundaries[city]:
        xs.extend([x for x, _ in seg])
        ys.extend([y for _, y in seg])
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pad_x = (xmax - xmin) * 0.045
    pad_y = (ymax - ymin) * 0.045
    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def make_transform(extent, rect, fixed_scale=None):
    xmin, xmax, ymin, ymax = extent
    scale = fixed_scale if fixed_scale is not None else min(rect[2] / (xmax - xmin), rect[3] / (ymax - ymin))
    map_w = (xmax - xmin) * scale
    map_h = (ymax - ymin) * scale
    ox = rect[0] + (rect[2] - map_w) / 2
    oy = rect[1] + (rect[3] - map_h) / 2

    def tr(x, y):
        return ox + (x - xmin) * scale, oy + (ymax - y) * scale

    return tr, scale


def draw_north(draw, x, y, size=34):
    outer = [(x, y - size), (x - size * 0.34, y + size * 0.62), (x, y + size * 0.28), (x + size * 0.34, y + size * 0.62)]
    inner = [(x, y - size * 0.76), (x - size * 0.14, y + size * 0.27), (x, y + size * 0.06), (x + size * 0.14, y + size * 0.27)]
    draw.polygon(outer, fill="white", outline=PAL["ink"])
    draw.polygon(inner, fill=PAL["ink"])
    text(draw, (x, y + size * 0.76), "N", F_TINY, PAL["ink"], anchor="ma")


def draw_scale(draw, rect, scale, length=3000):
    x0 = rect[0] + 50
    y0 = rect[1] + rect[3] - 52
    length_px = length * scale
    draw.line([(x0, y0), (x0 + length_px, y0)], fill=PAL["ink"], width=4)
    for frac, label in [(0, "0"), (0.5, "1.5"), (1, "3 km")]:
        xx = x0 + length_px * frac
        draw.line([(xx, y0 - 10), (xx, y0 + 10)], fill=PAL["ink"], width=2)
        text(draw, (xx, y0 + 16), label, F_TINY, PAL["ink"], anchor="ma")


def draw_points_rgba(base, records, tr):
    overlay = Image.new("RGBA", base.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(overlay)
    rgba = (11, 97, 114, 165)
    for row in records.itertuples(index=False):
        x, y = tr(row.x_m, row.y_m)
        d.ellipse((x - 2.4, y - 2.4, x + 2.4, y + 2.4), fill=rgba)
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


def save_all(img, stem):
    img.save(OUT / f"{stem}.png", dpi=(300, 300))
    img.save(OUT / f"{stem}.tiff", dpi=(600, 600))
    img.save(OUT / f"{stem}.pdf", "PDF", resolution=300)


def main():
    points = pd.read_csv(POINTS)
    # Baseline/endline OpenLocal retail records are shown together to summarize the study distribution.
    centres = points.groupby("city")[["lon", "lat"]].median().apply(tuple, axis=1).to_dict()
    boundaries = load_city_boundaries(centres)

    W, H = 1780, 900
    img = Image.new("RGB", (W, H), PAL["bg"])
    draw = ImageDraw.Draw(img)
    rects = {
        "Birmingham": (52, 72, 815, 720),
        "Liverpool": (912, 72, 815, 720),
    }
    extents = {city: extent_for(city, points, boundaries) for city in CITY_ORDER}
    common_scale = min(
        min(rects[city][2] / (extents[city][1] - extents[city][0]), rects[city][3] / (extents[city][3] - extents[city][2]))
        for city in CITY_ORDER
    )

    panel_info = {}
    for city in CITY_ORDER:
        rect = rects[city]
        draw.rectangle((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]), fill=PAL["panel"], outline=PAL["frame"], width=3)
        extent = extents[city]
        tr, scale = make_transform(extent, rect, fixed_scale=common_scale)
        panel_info[city] = (tr, scale)
        text(draw, (rect[0], rect[1] - 48), city, F_PANEL, PAL["ink"])

        for seg in boundaries[city]:
            pts = [tr(x, y) for x, y in simplify_segment(seg, 3)]
            if len(pts) > 1:
                draw.line(pts, fill=PAL["boundary"], width=2)

    for city in CITY_ORDER:
        tr, scale = panel_info[city]
        city_points = points.loc[points.city.eq(city), ["x_m", "y_m"]].dropna()
        img = draw_points_rgba(img, city_points, tr)
        draw = ImageDraw.Draw(img)
        rect = rects[city]
        for seg in boundaries[city]:
            pts = [tr(x, y) for x, y in simplify_segment(seg, 3)]
            if len(pts) > 1:
                draw.line(pts, fill=PAL["boundary"], width=2)
        draw_north(draw, rect[0] + 58, rect[1] + 82, 38)
        draw_scale(draw, rect, scale, length=3000)

    lx, ly = 870, 846
    draw.ellipse((lx, ly - 7, lx + 14, ly + 7), fill=PAL["point"])
    text(draw, (lx + 30, ly - 15), "OpenLocal retail property records", F_SMALL, PAL["ink"])
    draw.line([(lx + 455, ly), (lx + 555, ly)], fill=PAL["boundary"], width=3)
    text(draw, (lx + 575, ly - 15), "Local authority boundary", F_SMALL, PAL["ink"])

    save_all(img, "figure_3_2_study_area_boundaries_openlocal_retail_distribution")
    print(f"Wrote Figure 3.2 to {OUT}")


if __name__ == "__main__":
    main()

