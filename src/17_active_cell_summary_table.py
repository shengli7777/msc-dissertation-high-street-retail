from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "grid_250m_spatial_analysis"

SCALE = 3
W, H = 1280, 430

PAL = {
    "bg": "#FFFFFF",
    "ink": "#000000",
    "rule": "#D9E1E5",
    "rule_dark": "#AAB7BE",
}

ROWS = [
    ("Active cells", "1,556", "701"),
    ("Loss cells", "735 (47.2%)", "393 (56.1%)"),
    ("Growth cells", "222 (14.3%)", "85 (12.1%)"),
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


def text(draw, xy, s, size=23, fill=None, bold=False, anchor="la"):
    draw.text(
        (xy[0] * SCALE, xy[1] * SCALE),
        s,
        font=font(size, bold),
        fill=fill or PAL["ink"],
        anchor=anchor,
    )


def line(draw, pts, fill=None, width=1):
    draw.line([(x * SCALE, y * SCALE) for x, y in pts], fill=fill or PAL["rule"], width=width * SCALE)


def save_all(img, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / stem
    rgb = img.convert("RGB")
    rgb.save(base.with_suffix(".png"), dpi=(300, 300))
    rgb.save(base.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")
    rgb.save(base.with_suffix(".pdf"), "PDF", resolution=300)


def main():
    img = Image.new("RGBA", (W * SCALE, H * SCALE), PAL["bg"])
    draw = ImageDraw.Draw(img)

    left, top = 55, 36
    table_w = W - 110
    col_x = [left, left + 545, left + 940]

    text(draw, (left, top), "Summary of active 250 m grid cells", size=31, bold=True)

    header_y = top + 58
    text(draw, (col_x[0], header_y), "Metric", size=28, bold=True)
    text(draw, (col_x[1], header_y), "Birmingham", size=28, bold=True)
    text(draw, (col_x[2], header_y), "Liverpool", size=28, bold=True)
    line(draw, [(left, header_y + 34), (left + table_w, header_y + 34)], PAL["rule_dark"], width=1)

    y = header_y + 66
    for metric, birmingham, liverpool in ROWS:
        text(draw, (col_x[0], y), metric, size=28)
        text(draw, (col_x[1], y), birmingham, size=28)
        text(draw, (col_x[2], y), liverpool, size=28)
        line(draw, [(left, y + 36), (left + table_w, y + 36)], PAL["rule"], width=1)
        y += 58

    note_y = y + 20
    text(draw, (left, note_y), "Note: Loss cells include disappeared and declined cells. Growth cells include intensified and emerged cells.", size=18)
    text(draw, (left, note_y + 28), "Percentages are calculated within active 250 m grid cells for each city.", size=18)

    save_all(img, "table_active_cell_summary")


if __name__ == "__main__":
    main()

