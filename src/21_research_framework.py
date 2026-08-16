from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "research_framework_compact_outputs"
W, H = 1350, 1470
SCALE = 2

PAL = {
    "bg": "#FFFFFF",
    "ink": "#000000",
    "line": "#5C7480",
    "dash": "#8FB4C2",
    "light_arrow": "#A9C1CA",
    "box": "#C9DEE7",
    "box2": "#D7E8EE",
    "box3": "#E9F1F5",
}


def font(size, bold=False, serif=False):
    paths = (
        [
            "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
        ]
        if serif
        else [
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        ]
    )
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size * SCALE)
    return ImageFont.load_default()


def text(draw, x, y, s, size=18, fill=None, bold=False, anchor="mm", serif=False):
    draw.text(
        (x * SCALE, y * SCALE),
        s,
        font=font(size, bold, serif),
        fill=fill or PAL["ink"],
        anchor=anchor,
    )


def wrap(draw, s, max_width, size, bold=False):
    if "\n" in s:
        return s.split("\n")
    words = s.split()
    fnt = font(size, bold)
    lines, cur = [], ""
    for word in words:
        test = f"{cur} {word}".strip()
        if draw.textbbox((0, 0), test, font=fnt)[2] <= max_width * SCALE:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def multiline(draw, x, y, s, size=15, fill=None, bold=False, max_width=260):
    lines = wrap(draw, s, max_width, size, bold)
    fnt = font(size, bold)
    line_h = int(size * SCALE * 1.18)
    total = (len(lines) - 1) * line_h
    y0 = y * SCALE - total / 2
    for i, line in enumerate(lines):
        draw.text(
            (x * SCALE, y0 + i * line_h),
            line,
            font=fnt,
            fill=fill or PAL["ink"],
            anchor="mm",
        )


def rounded(draw, x, y, w, h, fill, outline="#AEC6D1", width=1, radius=4):
    draw.rounded_rectangle(
        [x * SCALE, y * SCALE, (x + w) * SCALE, (y + h) * SCALE],
        radius=radius * SCALE,
        fill=fill,
        outline=outline,
        width=width * SCALE,
    )


def dashed_rect(draw, x, y, w, h, label):
    dash, gap = 9 * SCALE, 7 * SCALE
    for (x1, y1), (x2, y2) in [
        ((x, y), (x + w, y)),
        ((x + w, y), (x + w, y + h)),
        ((x + w, y + h), (x, y + h)),
        ((x, y + h), (x, y)),
    ]:
        x1, y1, x2, y2 = x1 * SCALE, y1 * SCALE, x2 * SCALE, y2 * SCALE
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        for i in range(int(length // (dash + gap)) + 2):
            start = i * (dash + gap)
            end = min(start + dash, length)
            if start >= length:
                break
            t1, t2 = start / length, end / length
            xa, ya = x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1
            xb, yb = x1 + (x2 - x1) * t2, y1 + (y2 - y1) * t2
            draw.line([xa, ya, xb, yb], fill=PAL["dash"], width=2 * SCALE)
    text(draw, x + 14, y + 10, label, size=24, fill=PAL["ink"], bold=True, anchor="la")


def box(draw, x, y, w, h, label, fill=None, size=14, bold=True):
    rounded(draw, x, y, w, h, fill or PAL["box"], radius=4)
    multiline(draw, x + w / 2, y + h / 2 + 1, label, size=size, fill=PAL["ink"], bold=bold, max_width=w - 24)


def arrow(draw, x1, y1, x2, y2):
    draw.line([x1 * SCALE, y1 * SCALE, x2 * SCALE, y2 * SCALE], fill=PAL["line"], width=3 * SCALE)
    a = math.atan2(y2 - y1, x2 - x1)
    length, spread = 12, 0.48
    pts = [(x2 * SCALE, y2 * SCALE)]
    for sgn in (1, -1):
        aa = a + math.pi - sgn * spread
        pts.append(((x2 + length * math.cos(aa)) * SCALE, (y2 + length * math.sin(aa)) * SCALE))
    draw.polygon(pts, fill=PAL["line"])


def arrow_head(draw, x1, y1, x2, y2, fill):
    a = math.atan2(y2 - y1, x2 - x1)
    length, spread = 12, 0.48
    pts = [(x2 * SCALE, y2 * SCALE)]
    for sgn in (1, -1):
        aa = a + math.pi - sgn * spread
        pts.append(((x2 + length * math.cos(aa)) * SCALE, (y2 + length * math.sin(aa)) * SCALE))
    draw.polygon(pts, fill=fill)


def dashed_line(draw, x1, y1, x2, y2, fill, width=2):
    dash, gap = 8 * SCALE, 7 * SCALE
    x1, y1, x2, y2 = x1 * SCALE, y1 * SCALE, x2 * SCALE, y2 * SCALE
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    for i in range(int(length // (dash + gap)) + 2):
        start = i * (dash + gap)
        end = min(start + dash, length)
        if start >= length:
            break
        t1, t2 = start / length, end / length
        xa, ya = x1 + (x2 - x1) * t1, y1 + (y2 - y1) * t1
        xb, yb = x1 + (x2 - x1) * t2, y1 + (y2 - y1) * t2
        draw.line([xa, ya, xb, yb], fill=fill, width=width * SCALE)


def dashed_elbow_arrow(draw, points, fill=None):
    fill = fill or PAL["light_arrow"]
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        dashed_line(draw, x1, y1, x2, y2, fill, width=3)
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    dashed_line(draw, x1, y1, x2, y2, fill, width=3)
    arrow_head(draw, x1, y1, x2, y2, fill)


def elbow_arrow(draw, points):
    for (x1, y1), (x2, y2) in zip(points[:-2], points[1:-1]):
        draw.line([x1 * SCALE, y1 * SCALE, x2 * SCALE, y2 * SCALE], fill=PAL["line"], width=3 * SCALE)
    arrow(draw, points[-2][0], points[-2][1], points[-1][0], points[-1][1])


def draw_framework():
    img = Image.new("RGB", (W * SCALE, H * SCALE), PAL["bg"])
    draw = ImageDraw.Draw(img)

    dashed_rect(draw, 45, 48, 1260, 165, "Research Questions")
    dashed_rect(draw, 45, 240, 1260, 390, "Data and Indicator Construction")
    dashed_rect(draw, 45, 655, 1260, 355, "Analysis Strategy")
    dashed_rect(draw, 45, 1045, 1260, 380, "Interpretation and Policy")

    # Research questions.
    box(draw, 100, 88, 700, 82, "Empirical RQs\nRQ1 What changed? | RQ2 Where and how?", PAL["box"], size=24)
    box(draw, 840, 88, 400, 82, "Interpretive RQ\nRQ3 Planning implications?", PAL["box2"], size=23)
    arrow(draw, 450, 170, 675, 272)
    dashed_elbow_arrow(draw, [(1040, 170), (1275, 170), (1275, 1045)])

    # Data and indicators.
    box(draw, 335, 280, 680, 70, "OpenLocal Retail Property Dataset", PAL["box"], size=23)
    arrow(draw, 675, 350, 675, 375)
    box(draw, 335, 375, 680, 70, "Retail point records, 2016-2025\nBirmingham and Liverpool", PAL["box2"], size=21)
    arrow(draw, 675, 445, 675, 472)
    box(draw, 300, 472, 750, 68, "Indicator Construction", PAL["box"], size=23)
    arrow(draw, 675, 540, 675, 560)
    box(draw, 115, 560, 1120, 58, "Retail count | Known-case vacancy rate | Floor area | Retail mix | Shannon diversity", PAL["box3"], size=19)

    # Analysis strategy.
    arrow(draw, 675, 618, 675, 680)
    analysis = [
        (50, 192, "Temporal Analysis", "Overall retail change", 21, 21),
        (360, 502, "Functional Analysis", "Retail mix and diversity", 21, 21),
        (670, 812, "250 m Grid Analysis", "Spatial distribution\nand heterogeneity", 21, 20),
        (980, 1122, "HDBSCAN Analysis", "Cluster evolution\npersistence / emergence /\nrepositioning", 21, 20),
    ]
    for x, cx, title, result, title_size, result_size in analysis:
        box(draw, x, 695, 270, 74, title, PAL["box"], size=title_size)
        arrow(draw, 675, 680, cx, 695)
        arrow(draw, cx, 769, cx, 795)
        box(draw, x, 795, 285, 112, result, PAL["box2"], size=result_size)

    # Unified evidence flow into interpretation.
    for _, cx, *_ in analysis:
        arrow(draw, cx, 907, cx, 970)
    draw.line([190 * SCALE, 970 * SCALE, 1120 * SCALE, 970 * SCALE], fill=PAL["line"], width=3 * SCALE)
    arrow(draw, 675, 970, 675, 1095)

    # Interpretation and policy.
    box(draw, 290, 1095, 770, 76, "Interpretive Framework\nDecline | Spatial redistribution | Adaptation", PAL["box"], size=21)
    arrow(draw, 675, 1171, 675, 1215)
    box(draw, 125, 1215, 1100, 74, "Comparative interpretation\nBirmingham dispersed restructuring | Liverpool core-focused adjustment", PAL["box2"], size=20)
    arrow(draw, 675, 1289, 675, 1330)
    box(draw, 275, 1330, 800, 62, "Policy discussion\nplanning, regeneration, adaptive reuse", PAL["box3"], size=22)

    OUT.mkdir(parents=True, exist_ok=True)
    for stem in ["research_framework_large_boxes", "research_framework_compact_portrait", "research_framework_compact"]:
        img.save(OUT / f"{stem}.png")
        img.save(OUT / f"{stem}.tiff", dpi=(600, 600))
        img.save(OUT / f"{stem}.pdf", "PDF", resolution=300)
        old_svg = OUT / f"{stem}.svg"
        if old_svg.exists():
            old_svg.unlink()


if __name__ == "__main__":
    draw_framework()

