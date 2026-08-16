from pathlib import Path
import math
import struct
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "location_figures"
OUT.mkdir(parents=True, exist_ok=True)

LAD_SHP = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.shp"
LAD_DBF = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.dbf"

PAL = {
    "bg": "#FFFFFF",
    "land": "#F2F6F8",
    "boundary": "#D8E2E7",
    "coast": "#AAB6BE",
    "ink": "#1F2933",
    "muted": "#667085",
    "accent": "#D95F02",
    "birmingham": "#4C78A8",
    "liverpool": "#D95F02",
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


F_LABEL = font(31, True)
F_SMALL = font(20)
F_TINY = font(16)


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


def flatten(records):
    return [seg for rec_segments in records for seg in rec_segments]


def load_city_polygons():
    records = read_dbf_records(LAD_DBF)
    segments_by_record = read_shp_polygon_segments(LAD_SHP)
    city_polygons = {}
    for rec, segs in zip(records, segments_by_record):
        name = rec.get("LAD23NM")
        if name in ("Birmingham", "Liverpool"):
            city_polygons[name] = segs
    return city_polygons, segments_by_record


def simplify_segment(seg, step=8):
    if len(seg) <= step:
        return seg
    out = seg[::step]
    if out[-1] != seg[-1]:
        out.append(seg[-1])
    return out


def draw_north(draw, x, y, size=50):
    outer = [(x, y - size), (x - size * 0.34, y + size * 0.62), (x, y + size * 0.28), (x + size * 0.34, y + size * 0.62)]
    inner = [(x, y - size * 0.76), (x - size * 0.14, y + size * 0.27), (x, y + size * 0.06), (x + size * 0.14, y + size * 0.27)]
    draw.polygon(outer, fill="white", outline=PAL["ink"])
    draw.polygon(inner, fill=PAL["ink"])
    text(draw, (x, y + size * 0.76), "N", F_TINY, PAL["ink"], anchor="ma")


def draw_scale(draw, rect, transform, length_m=100000):
    x0 = rect[0] + 55
    y0 = rect[1] + rect[3] - 70
    x1, _ = transform(length_m, 0)
    x00, _ = transform(0, 0)
    length_px = abs(x1 - x00)
    draw.line([(x0, y0), (x0 + length_px, y0)], fill=PAL["ink"], width=5)
    for frac, label in [(0, "0"), (0.5, "50"), (1, "100 km")]:
        xx = x0 + length_px * frac
        draw.line([(xx, y0 - 12), (xx, y0 + 12)], fill=PAL["ink"], width=3)
        text(draw, (xx, y0 + 18), label, F_TINY, PAL["ink"], anchor="ma")


def save_all(img, stem):
    img.save(OUT / f"{stem}.png", dpi=(300, 300))
    img.save(OUT / f"{stem}.tiff", dpi=(600, 600))
    img.save(OUT / f"{stem}.pdf", "PDF", resolution=300)


def main():
    city_polygons, segments_by_record = load_city_polygons()
    segments = flatten(segments_by_record)
    city_xs = [x for segs in city_polygons.values() for seg in segs for x, _ in seg]
    city_ys = [y for segs in city_polygons.values() for seg in segs for _, y in seg]
    xmin, xmax = min(city_xs), max(city_xs)
    ymin, ymax = min(city_ys), max(city_ys)

    # Regional window around the two case-study cities; this keeps the
    # highlighted local authority boundaries legible and reduces unused UK base map.
    pad_x = (xmax - xmin) * 2.2
    pad_y = (ymax - ymin) * 1.8
    xmin -= pad_x
    xmax += pad_x
    ymin -= pad_y
    ymax += pad_y

    W, H = 1400, 1620
    img = Image.new("RGB", (W, H), PAL["bg"])
    draw = ImageDraw.Draw(img)
    rect = (55, 45, 1290, 1438)

    scale = min(rect[2] / (xmax - xmin), rect[3] / (ymax - ymin))
    map_w = (xmax - xmin) * scale
    map_h = (ymax - ymin) * scale
    ox = rect[0] + (rect[2] - map_w) / 2
    oy = rect[1] + (rect[3] - map_h) / 2

    def tr(x, y):
        return ox + (x - xmin) * scale, oy + (ymax - y) * scale

    # Fill local authority polygons first, then add very light LAD boundaries.
    # The boundaries provide GIS context without competing with the highlighted cities.
    for seg in segments:
        if not any(xmin <= x <= xmax and ymin <= y <= ymax for x, y in seg[::max(1, len(seg) // 20)]):
            continue
        pts = [tr(x, y) for x, y in simplify_segment(seg, 26)]
        if len(pts) > 2:
            draw.polygon(pts, fill=PAL["land"])
    for seg in segments:
        if not any(xmin <= x <= xmax and ymin <= y <= ymax for x, y in seg[::max(1, len(seg) // 20)]):
            continue
        pts = [tr(x, y) for x, y in simplify_segment(seg, 20)]
        if len(pts) > 1:
            draw.line(pts, fill=PAL["boundary"], width=1)

    city_centres = {}
    for name, segs in city_polygons.items():
        fill = PAL["birmingham"] if name == "Birmingham" else PAL["liverpool"]
        for seg in segs:
            pts = [tr(x, y) for x, y in simplify_segment(seg, 3)]
            if len(pts) > 2:
                draw.polygon(pts, fill=fill)
                draw.line(pts, fill=PAL["ink"], width=2)
        xs_city = [x for seg in segs for x, _ in seg]
        ys_city = [y for seg in segs for _, y in seg]
        city_centres[name] = tr((min(xs_city) + max(xs_city)) / 2, (min(ys_city) + max(ys_city)) / 2)

    label_offsets = {
        "Birmingham": (50, 8),
        "Liverpool": (50, -22),
    }
    for name, (x, y) in city_centres.items():
        dx, dy = label_offsets[name]
        text(draw, (x + dx, y + dy), name, F_LABEL, PAL["ink"])

    draw_north(draw, rect[0] + 62, rect[1] + 78, 48)
    draw_scale(draw, rect, tr, length_m=100000)

    save_all(img, "figure_3_1_location_birmingham_liverpool_uk")
    print(f"Wrote Figure 3.1 to {OUT}")


if __name__ == "__main__":
    main()




