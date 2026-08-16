from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "indicator_framework_outputs"

W, H = 1450, 1580
SCALE = 2

PAL = {
    "bg": "#FFFFFF",
    "ink": "#000000",
    "muted": "#5F7380",
    "line": "#5C7480",
    "dash": "#8FB4C2",
    "section_bg": "#FFFFFF",
    "input": "#D7E8EE",
    "method": "#E9F1F5",
    "method_dash": "#8FB4C2",
    "evidence": "#C9DEE7",
    "decline": "#D7E8EE",
    "relocation": "#D7E8EE",
    "adaptation": "#D7E8EE",
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


def text(draw, x, y, s, size=22, fill=None, bold=False, anchor="mm"):
    draw.text((x * SCALE, y * SCALE), s, font=font(size, bold), fill=fill or PAL["ink"], anchor=anchor)


def wrap(draw, s, max_width, size, bold=False):
    if "\n" in s:
        return s.split("\n")
    words, lines, cur = s.split(), [], ""
    fnt = font(size, bold)
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textbbox((0, 0), trial, font=fnt)[2] <= max_width * SCALE:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def multiline(draw, x, y, s, size=21, fill=None, bold=False, max_width=280, spacing=1.14):
    lines = wrap(draw, s, max_width, size, bold)
    fnt = font(size, bold)
    line_h = int(size * SCALE * spacing)
    y0 = y * SCALE - line_h * (len(lines) - 1) / 2
    for i, line in enumerate(lines):
        draw.text((x * SCALE, y0 + i * line_h), line, font=fnt, fill=fill or PAL["ink"], anchor="mm")


def rounded(draw, x, y, w, h, fill, outline="#A9BEC8", width=2, radius=8):
    draw.rounded_rectangle(
        [x * SCALE, y * SCALE, (x + w) * SCALE, (y + h) * SCALE],
        radius=radius * SCALE,
        fill=fill,
        outline=outline,
        width=width * SCALE,
    )


def dashed_line(draw, xy1, xy2, fill, width=2, dash=10, gap=7):
    x1, y1 = xy1
    x2, y2 = xy2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return
    for i in range(int(length // (dash + gap)) + 1):
        s = i * (dash + gap)
        e = min(s + dash, length)
        if s >= length:
            break
        xa, ya = x1 + dx * s / length, y1 + dy * s / length
        xb, yb = x1 + dx * e / length, y1 + dy * e / length
        draw.line([xa * SCALE, ya * SCALE, xb * SCALE, yb * SCALE], fill=fill, width=width * SCALE)


def dashed_rect(draw, x, y, w, h, label):
    rounded(draw, x, y, w, h, PAL["section_bg"], outline=None, width=0, radius=0)
    dashed_line(draw, (x, y), (x + w, y), PAL["dash"], width=2)
    dashed_line(draw, (x + w, y), (x + w, y + h), PAL["dash"], width=2)
    dashed_line(draw, (x + w, y + h), (x, y + h), PAL["dash"], width=2)
    dashed_line(draw, (x, y + h), (x, y), PAL["dash"], width=2)
    text(draw, x + 18, y + 18, label, size=25, fill=PAL["ink"], bold=True, anchor="la")


def box(draw, x, y, w, h, s, kind="input", size=21, bold=True, dashed=False):
    outline = "#AEC6D1" if not dashed else PAL["method_dash"]
    rounded(draw, x, y, w, h, PAL[kind], outline=outline, width=2, radius=8)
    if dashed:
        dashed_line(draw, (x + 8, y), (x + w - 8, y), PAL["method_dash"], width=2)
        dashed_line(draw, (x + w, y + 8), (x + w, y + h - 8), PAL["method_dash"], width=2)
        dashed_line(draw, (x + w - 8, y + h), (x + 8, y + h), PAL["method_dash"], width=2)
        dashed_line(draw, (x, y + h - 8), (x, y + 8), PAL["method_dash"], width=2)
    multiline(draw, x + w / 2, y + h / 2, s, size=size, fill=PAL["ink"], bold=bold, max_width=w - 36)


def arrow(draw, x1, y1, x2, y2, width=3):
    draw.line([x1 * SCALE, y1 * SCALE, x2 * SCALE, y2 * SCALE], fill=PAL["line"], width=width * SCALE)
    ang = math.atan2(y2 - y1, x2 - x1)
    length, spread = 17, 0.48
    pts = [(x2 * SCALE, y2 * SCALE)]
    for sign in (1, -1):
        aa = ang + math.pi - sign * spread
        pts.append(((x2 + length * math.cos(aa)) * SCALE, (y2 + length * math.sin(aa)) * SCALE))
    draw.polygon(pts, fill=PAL["line"])


def elbow(draw, points, width=3):
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        draw.line([x1 * SCALE, y1 * SCALE, x2 * SCALE, y2 * SCALE], fill=PAL["line"], width=width * SCALE)
    arrow(draw, points[-2][0], points[-2][1], points[-1][0], points[-1][1], width=width)


def draw_framework():
    img = Image.new("RGB", (W * SCALE, H * SCALE), PAL["bg"])
    draw = ImageDraw.Draw(img)

    dashed_rect(draw, 55, 35, 1340, 430, "Data and Operational Definition")
    dashed_rect(draw, 55, 500, 1340, 390, "Indicator Construction")
    dashed_rect(draw, 55, 925, 1340, 345, "Calculation Level and Spatial Evidence")
    dashed_rect(draw, 55, 1300, 1340, 230, "Interpretation")

    # Data and operational definition
    box(draw, 260, 75, 930, 76, "OpenLocal Retail Property Dataset", "input", size=28)
    box(draw, 260, 180, 930, 76, "VOA-related Classification Fields", "input", size=28)
    box(draw, 260, 285, 930, 76, "Spatial Property Records: Geometry | Area | Rateable Value", "input", size=26)
    arrow(draw, 725, 151, 725, 180)
    arrow(draw, 725, 256, 725, 285)
    box(draw, 105, 385, 560, 52, "Main definition: category_group = RETAIL", "method", size=23, dashed=True)
    box(draw, 785, 385, 560, 52, "Robustness: RETAIL_HIGH_STREET", "method", size=23, dashed=True)
    elbow(draw, [(610, 360), (385, 360), (385, 385)])
    elbow(draw, [(840, 360), (1065, 360), (1065, 385)])

    # Indicator construction
    arrow(draw, 725, 437, 725, 545)
    box(draw, 400, 545, 650, 70, "Retail Activity Indicators", "evidence", size=30)
    specs = [
        (100, 670, "Retail count\nscale of activity"),
        (355, 670, "Known-case\nvacancy rate\nvacancy pressure"),
        (610, 670, "Floor area\nretail space"),
        (865, 670, "Retail mix\nsubgroup composition"),
        (1120, 670, "Shannon diversity\nfunctional diversity"),
    ]
    for x, y, lab in specs:
        box(draw, x, y, 230, 92, lab, "input", size=22)
        arrow(draw, 725, 615, x + 115, y)
    box(
        draw,
        105,
        805,
        1240,
        62,
        "Indicators are calculated at both whole-city and 250 m grid levels where appropriate.",
        "method",
        size=22,
        dashed=True,
    )
    arrow(draw, 725, 762, 725, 805)

    # Calculation level and spatial evidence
    arrow(draw, 725, 867, 725, 960)
    box(draw, 75, 1000, 430, 100, "Whole-city indicators\nTemporal and functional change", "evidence", size=24)
    box(draw, 510, 1115, 430, 100, "250 m grid-based indicators\nSpatial distribution and heterogeneity", "evidence", size=23)
    box(draw, 945, 1000, 430, 100, "HDBSCAN cluster evidence\ncluster persistence, emergence,\ndisappearance, and repositioning", "evidence", size=22)
    arrow(draw, 725, 960, 290, 1000)
    arrow(draw, 725, 960, 725, 1115)
    arrow(draw, 725, 960, 1160, 1000)

    # Interpretation
    box(draw, 95, 1385, 350, 100, "Decline\nrecords, floor area,\nvacancy pressure", "decline", size=24)
    box(draw, 500, 1385, 450, 100, "Spatial redistribution\ngrid-level change and\ncluster repositioning", "relocation", size=23)
    box(draw, 1005, 1385, 350, 100, "Adaptation\nretail mix and\ndiversity change", "adaptation", size=24)
    arrow(draw, 290, 1100, 270, 1385)
    arrow(draw, 725, 1215, 725, 1385)
    arrow(draw, 1160, 1100, 725, 1385)
    elbow(draw, [(290, 1100), (290, 1285), (1180, 1285), (1180, 1385)])

    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "indicator_framework_large_boxes"
    img.save(base.with_suffix(".png"), dpi=(300, 300))
    img.save(base.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")
    img.save(base.with_suffix(".pdf"), "PDF", resolution=300)


if __name__ == "__main__":
    draw_framework()

