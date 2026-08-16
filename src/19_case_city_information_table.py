from pathlib import Path
import html

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "intro_figures_outputs"
SCALE_STATS = ROOT / "outputs" / "remade_progress_figures" / "remade_retail_scale_floor_area_stats.csv"
GRID_WIDE = ROOT / "outputs" / "grid_250m_spatial_analysis" / "grid_250m_change_metrics_wide.csv"

CITY_ORDER = ["Birmingham", "Liverpool"]
CITY_CODES = {"Birmingham": "E08000025", "Liverpool": "E08000012"}

PAL = {
    "bg": "#FFFFFF",
    "ink": "#111111",
    "muted": "#5F6670",
    "line": "#F58518",
    "header": "#F6FAFC",
    "stripe": "#FBFCFD",
    "accent": "#D95F02",
}


def font(size, bold=False, serif=False):
    paths = []
    if serif:
        paths = [
            "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/georgiab.ttf" if bold else "C:/Windows/Fonts/georgia.ttf",
        ]
    else:
        paths = [
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def wrap(draw, text, fnt, max_width):
    if "\n" in text:
        pieces = []
        for part in text.split("\n"):
            pieces.extend(wrap(draw, part, fnt, max_width))
        return pieces
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if draw.textbbox((0, 0), trial, font=fnt)[2] <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_wrapped(draw, xy, text, fnt, fill, max_width, line_gap=5, anchor="la"):
    x, y = xy
    lines = wrap(draw, str(text), fnt, max_width)
    line_h = fnt.size + line_gap
    if anchor.endswith("m"):
        y -= (len(lines) * line_h - line_gap) / 2
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_h), line, font=fnt, fill=fill, anchor="la")


def build_table():
    scale = pd.read_csv(SCALE_STATS)
    grid = pd.read_csv(GRID_WIDE)
    city_values = {}
    for city in CITY_ORDER:
        city_scale = scale[scale["geocode_name"].eq(city)]
        total_records = int(city_scale["retail_units"].sum())
        records_2016 = int(city_scale.loc[city_scale["year"].eq(2016), "retail_units"].iloc[0])
        records_2025 = int(city_scale.loc[city_scale["year"].eq(2025), "retail_units"].iloc[0])
        active_cells = int(grid[grid["city"].eq(city)].shape[0])
        city_values[city] = {
            "Local authority code": CITY_CODES[city],
            "Population": "1,144,900\n(Census 2021)" if city == "Birmingham" else "486,100\n(Census 2021)",
            "Median age": "34\n(Census 2021)" if city == "Birmingham" else "35\n(Census 2021)",
            "Employment / unemployment": (
                "Employment 62.1%;\nunemployment 7.7%\n(2023 profile)"
                if city == "Birmingham"
                else "Employment 67.5%;\nunemployment 7.0%\n(Jan-Dec 2023 profile)"
            ),
            "Study period": "2016-2025",
            "OpenLocal retail records": f"{total_records:,} total\n{records_2016:,} in 2016\n{records_2025:,} in 2025",
            "250 m grid coverage": f"{active_cells:,}\nactive grid cells",
        }

    indicators = [
        "Local authority code",
        "Population",
        "Median age",
        "Employment / unemployment",
        "Study period",
        "OpenLocal retail records",
        "250 m grid coverage",
    ]
    return pd.DataFrame({
        "Indicator": indicators,
        "Birmingham": [city_values["Birmingham"][i] for i in indicators],
        "Liverpool": [city_values["Liverpool"][i] for i in indicators],
    })


def draw_table_png(df):
    W, H = 2400, 1320
    margin_x = 150
    img = Image.new("RGB", (W, H), PAL["bg"])
    draw = ImageDraw.Draw(img)

    f_title = font(38, bold=False, serif=True)
    f_title_b = font(38, bold=True, serif=True)
    f_head = font(30, bold=True)
    f_cell = font(28)
    f_cell_b = font(28, bold=True)
    f_note = font(23)

    draw.text((margin_x, 95), "Table 1.", font=f_title_b, fill=PAL["accent"], anchor="la")
    draw.text((margin_x + 168, 95), "Summary statistics for selected case-study cities.",
              font=f_title, fill=PAL["ink"], anchor="la")
    draw.line([(margin_x, 170), (W - margin_x, 170)], fill=PAL["line"], width=4)

    table_x = margin_x
    table_y = 215
    table_w = W - 2 * margin_x
    col_w = [650, 710, 740]
    row_h = [86, 105, 115, 105, 160, 95, 145, 125]
    headers = list(df.columns)

    y = table_y
    for r, h in enumerate(row_h):
        fill = PAL["header"] if r == 0 else (PAL["stripe"] if r % 2 == 0 else PAL["bg"])
        draw.rectangle((table_x, y, table_x + table_w, y + h), fill=fill)
        draw.line([(table_x, y + h), (table_x + table_w, y + h)], fill="#DDE7EC", width=2)
        x = table_x
        for c, w in enumerate(col_w):
            draw.line([(x, y), (x, y + h)], fill="#E5EDF1", width=2)
            if r == 0:
                draw_wrapped(draw, (x + 24, y + h / 2), headers[c], f_head, PAL["ink"], w - 48, anchor="lm")
            else:
                val = df.iloc[r - 1, c]
                cell_font = f_cell_b if c == 0 else f_cell
                draw_wrapped(draw, (x + 24, y + h / 2), val, cell_font, PAL["ink"], w - 48, anchor="lm")
            x += w
        draw.line([(table_x + table_w, y), (table_x + table_w, y + h)], fill="#E5EDF1", width=2)
        y += h

    draw.line([(margin_x, y + 20), (W - margin_x, y + 20)], fill=PAL["line"], width=3)
    note = (
        "Source: ONS Census 2021 area profiles; Nomis labour market profiles, 2023; "
        "OpenLocal retail property dataset. OpenLocal coverage is computed from the dissertation retail analysis dataset."
    )
    draw_wrapped(draw, (margin_x, y + 70), note, f_note, PAL["muted"], W - 2 * margin_x)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "table1_case_city_basic_information.png"
    img.save(png, dpi=(300, 300))
    img.save(OUT / "table1_case_city_basic_information.tiff", dpi=(600, 600))
    img.save(OUT / "table1_case_city_basic_information.pdf", "PDF", resolution=300)
    return png


def write_svg(df):
    W, H = 2400, 1320
    margin_x = 150
    table_x = margin_x
    table_y = 215
    table_w = W - 2 * margin_x
    col_w = [650, 710, 740]
    row_h = [86, 105, 115, 105, 160, 95, 145, 125]
    y = table_y
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<rect width="{W}" height="{H}" fill="{PAL["bg"]}"/>',
        f'<text x="{margin_x}" y="122" font-family="Times New Roman, serif" font-size="38" font-weight="700" fill="{PAL["accent"]}">Table 1.</text>',
        f'<text x="{margin_x + 168}" y="122" font-family="Times New Roman, serif" font-size="38" fill="{PAL["ink"]}">Summary statistics for selected case-study cities.</text>',
        f'<line x1="{margin_x}" y1="170" x2="{W - margin_x}" y2="170" stroke="{PAL["line"]}" stroke-width="4"/>',
    ]
    headers = list(df.columns)
    for r, h in enumerate(row_h):
        fill = PAL["header"] if r == 0 else (PAL["stripe"] if r % 2 == 0 else PAL["bg"])
        parts.append(f'<rect x="{table_x}" y="{y}" width="{table_w}" height="{h}" fill="{fill}"/>')
        parts.append(f'<line x1="{table_x}" y1="{y+h}" x2="{table_x+table_w}" y2="{y+h}" stroke="#DDE7EC" stroke-width="2"/>')
        x = table_x
        for c, w in enumerate(col_w):
            parts.append(f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+h}" stroke="#E5EDF1" stroke-width="2"/>')
            raw = headers[c] if r == 0 else str(df.iloc[r - 1, c])
            escaped = html.escape(raw).replace("\n", " / ")
            weight = "700" if (r == 0 or (r > 0 and c == 0)) else "400"
            size = 30 if r == 0 else 28
            parts.append(
                f'<text x="{x+24}" y="{y+h/2+10}" font-family="Arial, sans-serif" '
                f'font-size="{size}" font-weight="{weight}" fill="{PAL["ink"]}">{escaped}</text>'
            )
            x += w
        parts.append(f'<line x1="{table_x+table_w}" y1="{y}" x2="{table_x+table_w}" y2="{y+h}" stroke="#E5EDF1" stroke-width="2"/>')
        y += h
    parts.append(f'<line x1="{margin_x}" y1="{y+20}" x2="{W-margin_x}" y2="{y+20}" stroke="{PAL["line"]}" stroke-width="3"/>')
    note = html.escape(
        "Source: ONS Census 2021 area profiles; Nomis labour market profiles, 2023; "
        "OpenLocal retail property dataset. OpenLocal coverage is computed from the dissertation retail analysis dataset."
    )
    parts.append(f'<text x="{margin_x}" y="{y+85}" font-family="Arial, sans-serif" font-size="23" fill="{PAL["muted"]}">{note}</text>')
    parts.append("</svg>")
    (OUT / "table1_case_city_basic_information.svg").write_text("\n".join(parts), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_table()
    df.to_csv(OUT / "table1_case_city_basic_information.csv", index=False, encoding="utf-8-sig")
    draw_table_png(df)
    write_svg(df)


if __name__ == "__main__":
    main()

