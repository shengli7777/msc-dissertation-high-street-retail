from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "grid_250m_spatial_analysis"

SCALE = 3
W, H = 1650, 730

PALETTE = {
    "bg": "#FFFFFF",
    "ink": "#000000",
    "muted": "#596770",
    "rule": "#D9E1E5",
    "rule_dark": "#AAB7BE",
    "header_bg": "#FFFFFF",
}

ROWS = [
    (
        "Disappeared",
        "Records present in 2016 but absent in 2025",
        "Retail activity disappeared from the cell",
    ),
    (
        "Declined",
        "Records present in both years, but fewer records in 2025",
        "Retail activity weakened",
    ),
    (
        "Persistent",
        "Records present in both years, with no change in count",
        "Retail activity remained stable",
    ),
    (
        "Intensified",
        "Records present in both years, with more records in 2025",
        "Retail activity strengthened",
    ),
    (
        "Emerged",
        "No records in 2016 but records present in 2025",
        "New retail activity appeared",
    ),
]


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size * SCALE)
    return ImageFont.load_default()


def draw_text(draw, xy, content, size=25, bold=False, fill=None, anchor="la"):
    draw.text(
        (xy[0] * SCALE, xy[1] * SCALE),
        content,
        font=font(size, bold),
        fill=fill or PALETTE["ink"],
        anchor=anchor,
    )


def text_size(draw, content, size=25, bold=False):
    box = draw.textbbox((0, 0), content, font=font(size, bold))
    return (box[2] - box[0]) / SCALE, (box[3] - box[1]) / SCALE


def wrap_text(draw, content, max_width, size=25, bold=False):
    words = content.split()
    lines = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if text_size(draw, trial, size, bold)[0] <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(draw, xy, content, max_width, size=25, bold=False, fill=None, line_gap=11):
    lines = wrap_text(draw, content, max_width, size, bold)
    x, y = xy
    line_h = text_size(draw, "Ag", size, bold)[1] + line_gap
    for idx, line in enumerate(lines):
        draw_text(draw, (x, y + idx * line_h), line, size=size, bold=bold, fill=fill)
    return len(lines) * line_h


def rect(draw, xy, fill=None, outline=None, width=1):
    draw.rectangle([v * SCALE for v in xy], fill=fill, outline=outline, width=width * SCALE)


def line(draw, xy, fill=None, width=1):
    draw.line([(x * SCALE, y * SCALE) for x, y in xy], fill=fill or PALETTE["rule"], width=width * SCALE)


def save_all(img, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / stem
    rgb = img.convert("RGB")
    rgb.save(base.with_suffix(".png"), dpi=(300, 300))
    rgb.save(base.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")
    rgb.save(base.with_suffix(".pdf"), "PDF", resolution=300)


def main():
    img = Image.new("RGBA", (W * SCALE, H * SCALE), PALETTE["bg"])
    draw = ImageDraw.Draw(img)

    left, top = 50, 36
    table_w = W - 100
    header_h = 64
    row_h = 106
    col_w = [290, 620, 640]
    x = [left, left + col_w[0], left + col_w[0] + col_w[1], left + table_w]

    rect(draw, (left, top, left + table_w, top + header_h), fill=PALETTE["header_bg"])
    line(draw, [(left, top), (left + table_w, top)], PALETTE["rule_dark"], width=2)
    line(draw, [(left, top + header_h), (left + table_w, top + header_h)], PALETTE["rule_dark"], width=1)

    headers = ["Grid-cell status", "Definition", "Interpretation"]
    for i, header in enumerate(headers):
        draw_text(draw, (x[i] + 24, top + 20), header, size=30, bold=True)

    y = top + header_h
    for status, definition, interpretation in ROWS:
        line(draw, [(left, y), (left + table_w, y)], PALETTE["rule"], width=1)
        draw_text(draw, (x[0] + 24, y + 31), status, size=28, bold=True)
        draw_wrapped(draw, (x[1] + 24, y + 24), definition, col_w[1] - 58, size=27, line_gap=8)
        draw_wrapped(draw, (x[2] + 24, y + 24), interpretation, col_w[2] - 58, size=27, line_gap=8)
        y += row_h

    line(draw, [(left, y), (left + table_w, y)], PALETTE["rule_dark"], width=2)

    note_y = y + 27
    draw_text(draw, (left, note_y), "Note:", size=22, bold=True, fill=PALETTE["ink"])
    note_x = left + 78
    note = "Grid-cell status is classified by comparing OpenLocal retail record counts within each 250 m grid cell between 2016 and 2025."
    draw_text(draw, (note_x, note_y), note, size=22, fill=PALETTE["ink"])

    save_all(img, "table_grid_cell_status_definitions")


if __name__ == "__main__":
    main()

