from pathlib import Path
import math
import struct

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "grid_250m_spatial_analysis"
LAD_SHP = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.shp"
LAD_DBF = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.dbf"

YEARS = [2016, 2019, 2022, 2025]
CITIES = ["Birmingham", "Liverpool"]
GRID_SIZE = 250

SCALE = 2
W, H = 1580, 1720

PAL = {
    "bg": "#FFFFFF",
    "ink": "#25313A",
    "muted": "#65727A",
    "boundary": "#7A8791",
    "panel_bg": "#FBFCFD",
    "panel_line": "#E6EEF2",
    "dark": "#17212B",
    "missing": "#DDE5E8",
}

BINS = [
    (1, 25_000, "拢1-25k", "#EEF5F2"),
    (25_000, 100_000, "拢25k-100k", "#CDE5DC"),
    (100_000, 250_000, "拢100k-250k", "#9BC9BC"),
    (250_000, 500_000, "拢250k-500k", "#63A99F"),
    (500_000, 1_000_000, "拢500k-1m", "#2F7F88"),
    (1_000_000, math.inf, ">拢1m", "#3F4D7D"),
]


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


def load_boundaries():
    records = read_dbf_records(LAD_DBF)
    segments = read_shp_polygon_segments(LAD_SHP)
    out = {}
    for rec, segs in zip(records, segments):
        city = rec.get("LAD23NM")
        if city in CITIES:
            out[city] = segs
    return out


def value_color(value):
    if pd.isna(value) or value <= 0:
        return PAL["missing"]
    for low, high, _, color in BINS:
        if low <= value <= high:
            return color
    return BINS[-1][3]


def data_extent(grid, boundary):
    xs = grid["grid_x"].dropna().tolist()
    xs.extend((grid["grid_x"] + GRID_SIZE).dropna().tolist())
    ys = grid["grid_y"].dropna().tolist()
    ys.extend((grid["grid_y"] + GRID_SIZE).dropna().tolist())
    for seg in boundary:
        xs.extend([x for x, _ in seg])
        ys.extend([y for _, y in seg])
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    pad_x = (xmax - xmin) * 0.04
    pad_y = (ymax - ymin) * 0.04
    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def transform_for_extent(extent, panel):
    xmin, xmax, ymin, ymax = extent
    x0, y0, w, h = panel
    s = min(w / (xmax - xmin), h / (ymax - ymin))
    used_w = (xmax - xmin) * s
    used_h = (ymax - ymin) * s
    ox = x0 + (w - used_w) / 2
    oy = y0 + (h - used_h) / 2

    def tr(x, y):
        return ox + (x - xmin) * s, oy + used_h - (y - ymin) * s

    return tr, s


def draw_boundary(draw, boundary, tr):
    for seg in boundary:
        pts = [tr(x, y) for x, y in seg]
        if len(pts) > 1:
            line(draw, pts, PAL["boundary"], width=2)


def draw_grid_cells(draw, grid, city, year, tr):
    cells = grid[grid["city"].eq(city) & grid["year"].eq(year)].sort_values("rateable_value_total", na_position="first")
    for row in cells.itertuples():
        x1, y1 = tr(row.grid_x, row.grid_y + GRID_SIZE)
        x2, y2 = tr(row.grid_x + GRID_SIZE, row.grid_y)
        rect(draw, (x1, y1, x2, y2), value_color(row.rateable_value_total), outline=None)


def draw_north(draw, x, y, size=32):
    pts = [(x, y - size), (x - size * 0.35, y + size * 0.45), (x, y + size * 0.12), (x + size * 0.35, y + size * 0.45)]
    draw.polygon([(px * SCALE, py * SCALE) for px, py in pts], fill="white", outline=PAL["dark"])
    inner = [(x, y - size * 0.70), (x - size * 0.10, y + size * 0.05), (x, y - size * 0.10), (x + size * 0.10, y + size * 0.05)]
    draw.polygon([(px * SCALE, py * SCALE) for px, py in inner], fill=PAL["dark"])
    text(draw, (x, y + size * 0.68), "N", size=13, fill=PAL["dark"], bold=True, anchor="mm")


def draw_scale(draw, x, y, pixel_scale, length_m=6000):
    length_px = length_m * pixel_scale
    line(draw, [(x, y), (x + length_px, y)], PAL["dark"], width=3)
    for xx in [x, x + length_px / 2, x + length_px]:
        line(draw, [(xx, y - 6), (xx, y + 6)], PAL["dark"], width=2)
    text(draw, (x, y + 15), "0", size=11, fill=PAL["dark"], anchor="ma")
    text(draw, (x + length_px / 2, y + 15), "3", size=11, fill=PAL["dark"], anchor="ma")
    text(draw, (x + length_px, y + 15), "6 km", size=11, fill=PAL["dark"], anchor="ma")


def draw_legend(draw, x, y):
    text(draw, (x, y), "Total rateable value", size=18, bold=True)
    text(draw, (x, y + 21), "per 250 m grid", size=18, bold=True)
    for i, (_, _, label, color) in enumerate(BINS):
        yy = y + 56 + i * 31
        rect(draw, (x, yy - 11, x + 24, yy + 11), color, outline="#FFFFFF", width=1)
        text(draw, (x + 34, yy - 12), label, size=15)
    yy = y + 256
    rect(draw, (x, yy - 11, x + 24, yy + 11), PAL["missing"], outline="#FFFFFF", width=1)
    text(draw, (x + 34, yy - 12), "No rateable value", size=15)
    yy += 45
    line(draw, [(x, yy), (x + 48, yy)], PAL["boundary"], width=2)
    text(draw, (x + 60, yy - 12), "Local authority boundary", size=15)


def main():
    grid = pd.read_csv(OUT / "grid_250m_rateable_value_selected_years.csv")
    boundaries = load_boundaries()
    extents = {city: data_extent(grid[grid["city"].eq(city)], boundaries[city]) for city in CITIES}

    img = Image.new("RGBA", (W * SCALE, H * SCALE), PAL["bg"])
    draw = ImageDraw.Draw(img)

    x_positions = {"Birmingham": 120, "Liverpool": 695}
    row_tops = {year: 105 + idx * 390 for idx, year in enumerate(YEARS)}
    panel_w, panel_h = 545, 365
    scale_ref = None

    text(draw, (x_positions["Birmingham"] + panel_w / 2, 55), "Birmingham", size=27, bold=True, anchor="mm")
    text(draw, (x_positions["Liverpool"] + panel_w / 2, 55), "Liverpool", size=27, bold=True, anchor="mm")

    for year in YEARS:
        y0 = row_tops[year]
        text(draw, (75, y0 + panel_h / 2), str(year), size=24, bold=True, anchor="mm")
        for city in CITIES:
            panel = (x_positions[city], y0, panel_w, panel_h)
            rect(draw, (panel[0], panel[1], panel[0] + panel[2], panel[1] + panel[3]), PAL["panel_bg"], outline=PAL["panel_line"])
            tr, s = transform_for_extent(extents[city], panel)
            if city == "Birmingham" and year == 2025:
                scale_ref = s
            draw_grid_cells(draw, grid, city, year, tr)
            draw_boundary(draw, boundaries[city], tr)

    draw_north(draw, 50, 80, size=34)
    draw_scale(draw, 52, 1665, scale_ref, length_m=6000)
    draw_legend(draw, 1265, 1350)
    save_all(img, "figure_rateable_value_indicator_map_2016_2025")


if __name__ == "__main__":
    main()

