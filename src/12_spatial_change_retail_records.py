from pathlib import Path
import math
import struct

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "grid_250m_spatial_analysis"
WIDE_CSV = OUT / "grid_250m_change_metrics_wide.csv"
LAD_SHP = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.shp"
LAD_DBF = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.dbf"

GRID_SIZE = 250
SCALE = 2
W, H = 1650, 2350
CITY_ORDER = ["Birmingham", "Liverpool"]

PAL = {
    "bg": "#FFFFFF",
    "ink": "#25313A",
    "muted": "#65727A",
    "boundary": "#A9B4BA",
    "panel_bg": "#FFFFFF",
    "panel_line": "#111111",
    "dark": "#17212B",
}

STATUS = [
    ("disappeared", "Disappeared", "#7A3B4A"),
    ("declined", "Declined", "#D18A62"),
    ("persistent", "Persistent", "#BFC5CA"),
    ("intensified", "Intensified", "#4C78A8"),
    ("emerged", "Emerged", "#E1A72E"),
]
STATUS_COLOR = {key: color for key, _, color in STATUS}


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
    draw.line([(x * SCALE, y * SCALE) for x, y in pts], fill=fill, width=max(1, round(width * SCALE)), joint="curve")


def rect(draw, xy, fill, outline=None, width=1):
    draw.rectangle([v * SCALE for v in xy], fill=fill, outline=outline, width=max(1, round(width * SCALE)))


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
        if city in CITY_ORDER:
            out[city] = segs
    return out


def data_extent(cells, boundary):
    xs = cells["grid_x"].dropna().tolist()
    xs.extend((cells["grid_x"] + GRID_SIZE).dropna().tolist())
    ys = cells["grid_y"].dropna().tolist()
    ys.extend((cells["grid_y"] + GRID_SIZE).dropna().tolist())
    for seg in boundary:
        xs.extend([x for x, _ in seg])
        ys.extend([y for _, y in seg])
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    pad_x = (xmax - xmin) * 0.035
    pad_y = (ymax - ymin) * 0.035
    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def transform_for_extent(extent, panel, fixed_scale=None, align_x=0.5):
    xmin, xmax, ymin, ymax = extent
    x0, y0, w, h = panel
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
            line(draw, pts, PAL["boundary"], width=1.2)


def draw_cells(draw, cells, tr):
    for row in cells.sort_values("cell_status").itertuples():
        color = STATUS_COLOR.get(row.cell_status, "#CCCCCC")
        x1, y1 = tr(row.grid_x, row.grid_y + GRID_SIZE)
        x2, y2 = tr(row.grid_x + GRID_SIZE, row.grid_y)
        rect(draw, (x1, y1, x2, y2), color, outline="#FFFFFF", width=0.5)


def draw_north(draw, x, y, size=42):
    pts = [(x, y - size), (x - size * 0.35, y + size * 0.45), (x, y + size * 0.12), (x + size * 0.35, y + size * 0.45)]
    draw.polygon([(px * SCALE, py * SCALE) for px, py in pts], fill="white", outline=PAL["dark"])
    inner = [(x, y - size * 0.70), (x - size * 0.10, y + size * 0.05), (x, y - size * 0.10), (x + size * 0.10, y + size * 0.05)]
    draw.polygon([(px * SCALE, py * SCALE) for px, py in inner], fill=PAL["dark"])
    text(draw, (x, y + size * 0.68), "N", size=20, fill=PAL["dark"], bold=True, anchor="mm")


def draw_scale(draw, x, y, pixel_scale, length_m=3000):
    length_px = length_m * pixel_scale
    line(draw, [(x, y), (x + length_px, y)], PAL["dark"], width=4)
    for xx in [x, x + length_px / 2, x + length_px]:
        line(draw, [(xx, y - 9), (xx, y + 9)], PAL["dark"], width=2.5)
    text(draw, (x, y + 28), "0", size=18, fill=PAL["dark"], anchor="ma")
    text(draw, (x + length_px / 2, y + 28), "1.5", size=18, fill=PAL["dark"], anchor="ma")
    text(draw, (x + length_px, y + 28), "3 km", size=18, fill=PAL["dark"], anchor="ma")


def draw_legend(draw, x, y):
    text(draw, (x, y), "Grid-cell status", size=26, bold=True)
    for i, (_, label, color) in enumerate(STATUS):
        yy = y + 48 + i * 42
        rect(draw, (x, yy - 15, x + 34, yy + 17), color, outline="#FFFFFF", width=0.5)
        text(draw, (x + 48, yy - 17), label, size=23)
    yy = y + 285
    line(draw, [(x, yy), (x + 76, yy)], PAL["boundary"], width=1.5)
    text(draw, (x + 92, yy - 17), "Local authority boundary", size=22)


def main():
    wide = pd.read_csv(WIDE_CSV)
    boundaries = load_boundaries()
    panels = {
        "Birmingham": (55, 105, 1540, 1010),
        "Liverpool": (55, 1260, 1540, 1010),
    }
    extents = {}
    common_scale = None
    for city, panel in panels.items():
        cells = wide[wide["city"].eq(city)].copy()
        boundary = boundaries.get(city, [])
        extent = data_extent(cells, boundary)
        extents[city] = extent
        xmin, xmax, ymin, ymax = extent
        city_scale = min(panel[2] / (xmax - xmin), panel[3] / (ymax - ymin))
        common_scale = city_scale if common_scale is None else min(common_scale, city_scale)

    img = Image.new("RGBA", (W * SCALE, H * SCALE), PAL["bg"])
    draw = ImageDraw.Draw(img)

    for city, panel in panels.items():
        cells = wide[wide["city"].eq(city)].copy()
        boundary = boundaries.get(city, [])
        tr, panel_scale = transform_for_extent(extents[city], panel, fixed_scale=common_scale, align_x=0.31)
        rect(draw, (panel[0], panel[1], panel[0] + panel[2], panel[1] + panel[3]), PAL["panel_bg"], outline=PAL["panel_line"], width=1.4)
        text(draw, (panel[0] + panel[2] / 2, panel[1] - 38), city, size=40, bold=True, anchor="mm")
        draw_cells(draw, cells, tr)
        draw_boundary(draw, boundary, tr)
        draw_north(draw, panel[0] + 72, panel[1] + 88, size=48)
        draw_scale(draw, panel[0] + 58, panel[1] + panel[3] - 70, panel_scale, length_m=3000)
        draw_legend(draw, panel[0] + panel[2] - 390, panel[1] + panel[3] - 390)
    save_all(img, "figure_spatial_change_retail_records_status_vertical")

    single_panel = (55, 95, 1540, 950)
    for city in CITY_ORDER:
        single_img = Image.new("RGBA", (1650 * SCALE, 1175 * SCALE), PAL["bg"])
        single_draw = ImageDraw.Draw(single_img)
        cells = wide[wide["city"].eq(city)].copy()
        boundary = boundaries.get(city, [])
        tr, panel_scale = transform_for_extent(extents[city], single_panel, fixed_scale=common_scale, align_x=0.31)
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
        text(single_draw, (single_panel[0] + single_panel[2] / 2, single_panel[1] - 34), city, size=40, bold=True, anchor="mm")
        draw_cells(single_draw, cells, tr)
        draw_boundary(single_draw, boundary, tr)
        draw_north(single_draw, single_panel[0] + 72, single_panel[1] + 88, size=48)
        draw_scale(single_draw, single_panel[0] + 58, single_panel[1] + single_panel[3] - 70, panel_scale, length_m=3000)
        draw_legend(single_draw, single_panel[0] + single_panel[2] - 390, single_panel[1] + single_panel[3] - 385)
        save_all(single_img, f"figure_spatial_change_retail_records_status_{city.lower()}")


if __name__ == "__main__":
    main()

