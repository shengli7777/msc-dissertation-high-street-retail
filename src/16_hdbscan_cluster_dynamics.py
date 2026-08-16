from pathlib import Path
import math
import struct
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs'
OUT = ROOT / 'work' / 'cluster_figures_4_14_4_17'
OUT.mkdir(parents=True, exist_ok=True)
LAD_SHP = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.shp"
LAD_DBF = ROOT / "data" / "raw" / "LAD_MAY_2023_UK_BGC_V2.dbf"

ASSIGN = pd.read_csv(DATA / 'hdbscan_retail_clusters_assignments.csv')
SUMMARY = pd.read_csv(DATA / 'hdbscan_retail_clusters_summary.csv')
MOVES = pd.read_csv(DATA / 'hdbscan_retail_clusters_movements.csv')

CITY_ORDER = ['Birmingham', 'Liverpool']
LIVERPOOL_HALF_PAGE_SCALE = 0.0685
PAL = {
    'bg': '#FFFFFF', 'ink': '#1F2933', 'muted': '#667085', 'grid': '#E6EEF2',
    'line': '#6E8D99', 'accent': '#D95F02', 'bham': '#4C78A8', 'liv': '#F58518',
    'early': '#4C78A8', 'late': '#D95F02', 'persistent': '#4C78A8',
    'emerged': '#2A9D8F', 'disappeared': '#C44E52', 'neutral': '#B8C7D0',
    'point': '#AEB8C2', 'boundary': '#7A8791', 'dark': '#111827'
}
CITY_COLOR = {'Birmingham': PAL['bham'], 'Liverpool': PAL['liv']}
STATUS_LABEL = {
    'persistent_or_relocated': 'Persistent / matched',
    'emerged': 'Emerged',
    'disappeared': 'Disappeared'
}
STATUS_COLOR = {
    'persistent_or_relocated': PAL['persistent'],
    'emerged': PAL['emerged'],
    'disappeared': PAL['disappeared']
}


def font(size, bold=False, serif=False):
    if serif:
        paths = ['C:/Windows/Fonts/timesbd.ttf' if bold else 'C:/Windows/Fonts/times.ttf',
                 'C:/Windows/Fonts/georgiab.ttf' if bold else 'C:/Windows/Fonts/georgia.ttf']
    else:
        paths = ['C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',
                 'C:/Windows/Fonts/calibrib.ttf' if bold else 'C:/Windows/Fonts/calibri.ttf']
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

F_TITLE = font(42, bold=True, serif=True)
F_SUB = font(24)
F_PANEL = font(34, bold=True)
F_HALF_TITLE = font(58, bold=True)
F_HALF_LEG = font(31)
F_HALF_LEG_BOLD = font(35, bold=True)
F_HALF_NOTE = font(30, bold=True)
F_HALF_DISTANCE = font(31, bold=True)
F_LABEL = font(26)
F_SMALL = font(22)
F_TINY = font(18)
F_BOLD = font(26, bold=True)
F_DISTANCE = font(24, bold=True)


def text(draw, xy, s, fnt, fill=PAL['ink'], anchor='la'):
    draw.text(xy, s, font=fnt, fill=fill, anchor=anchor)


def rotated_text(img, xy, s, fnt, fill=PAL['ink'], angle=90, anchor='mm'):
    bbox = ImageDraw.Draw(Image.new('RGBA', (1, 1))).textbbox((0, 0), s, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = 8
    layer = Image.new('RGBA', (tw + pad * 2, th + pad * 2), (255, 255, 255, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((pad - bbox[0], pad - bbox[1]), s, font=fnt, fill=fill)
    rotated = layer.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    x, y = xy
    if anchor == 'mm':
        x -= rotated.width / 2
        y -= rotated.height / 2
    if img.mode == 'RGBA':
        img.alpha_composite(rotated, (round(x), round(y)))
    else:
        img.paste(rotated.convert('RGB'), (round(x), round(y)), rotated)


def save_all(img, stem):
    out = img.convert('RGB') if img.mode == 'RGBA' else img
    out.save(OUT / f'{stem}.png', dpi=(300, 300))
    out.save(OUT / f'{stem}.tiff', dpi=(600, 600))
    out.save(OUT / f'{stem}.pdf', 'PDF', resolution=300)


def title_block(draw, fig_no, title, subtitle=None):
    return


def source_note(draw, y):
    return


def read_dbf_records(path):
    with open(path, 'rb') as f:
        header = f.read(32)
        n_records = struct.unpack('<I', header[4:8])[0]
        header_len = struct.unpack('<H', header[8:10])[0]
        record_len = struct.unpack('<H', header[10:12])[0]
        fields = []
        while f.tell() < header_len - 1:
            desc = f.read(32)
            if not desc or desc[0] == 0x0D:
                break
            name = desc[:11].split(b'\x00', 1)[0].decode('ascii', errors='ignore')
            length = desc[16]
            fields.append((name, length))
        f.seek(header_len)
        records = []
        for _ in range(n_records):
            raw = f.read(record_len)
            if not raw or raw[:1] == b'*':
                continue
            pos = 1
            rec = {}
            for name, length in fields:
                rec[name] = raw[pos:pos+length].decode('latin1', errors='ignore').strip()
                pos += length
            records.append(rec)
    return records


def read_shp_polygon_segments(path):
    segments_by_record = []
    with open(path, 'rb') as f:
        f.seek(100)
        while True:
            rec_header = f.read(8)
            if len(rec_header) < 8:
                break
            content_len_words = struct.unpack('>i', rec_header[4:8])[0]
            content = f.read(content_len_words * 2)
            if len(content) < 44:
                segments_by_record.append([])
                continue
            shape_type = struct.unpack('<i', content[:4])[0]
            if shape_type not in (5, 15, 25):
                segments_by_record.append([])
                continue
            num_parts, num_points = struct.unpack('<2i', content[36:44])
            parts_start = 44
            parts = list(struct.unpack('<' + 'i' * num_parts, content[parts_start:parts_start + 4 * num_parts]))
            points_start = parts_start + 4 * num_parts
            points = []
            for i in range(num_points):
                x, y = struct.unpack('<2d', content[points_start + i * 16:points_start + (i + 1) * 16])
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
        ma = (1 + n + 5/4*n**2 + 5/4*n**3) * (lat - lat0)
        mb = (3*n + 3*n**2 + 21/8*n**3) * math.sin(lat - lat0) * math.cos(lat + lat0)
        mc = (15/8*n**2 + 15/8*n**3) * math.sin(2*(lat - lat0)) * math.cos(2*(lat + lat0))
        md = 35/24*n**3 * math.sin(3*(lat - lat0)) * math.cos(3*(lat + lat0))
        m = b * f0 * (ma - mb + mc - md)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    tan_lat = math.tan(lat)
    nu = a * f0 / math.sqrt(1 - e2 * sin_lat**2)
    rho = a * f0 * (1 - e2) / (1 - e2 * sin_lat**2) ** 1.5
    eta2 = nu / rho - 1
    d_e = easting - e0
    vii = tan_lat / (2 * rho * nu)
    viii = tan_lat / (24 * rho * nu**3) * (5 + 3*tan_lat**2 + eta2 - 9*tan_lat**2*eta2)
    ix = tan_lat / (720 * rho * nu**5) * (61 + 90*tan_lat**2 + 45*tan_lat**4)
    x = 1 / (cos_lat * nu)
    xi = 1 / (6 * cos_lat * nu**3) * (nu / rho + 2*tan_lat**2)
    xii = 1 / (120 * cos_lat * nu**5) * (5 + 28*tan_lat**2 + 24*tan_lat**4)
    xiia = 1 / (5040 * cos_lat * nu**7) * (61 + 662*tan_lat**2 + 1320*tan_lat**4 + 720*tan_lat**6)
    lat = lat - vii*d_e**2 + viii*d_e**4 - ix*d_e**6
    lon = lon0 + x*d_e - xi*d_e**3 + xii*d_e**5 - xiia*d_e**7
    return math.degrees(lon), math.degrees(lat)


def relative_xy(city, lon, lat):
    city_points = ASSIGN[ASSIGN.city.eq(city)]
    lon0 = city_points['lon'].median()
    lat0 = city_points['lat'].median()
    x = (lon - lon0) * 111_320.0 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110_540.0
    return x, y


def load_lad_boundaries():
    if not (LAD_SHP.exists() and LAD_DBF.exists()):
        return {}
    records = read_dbf_records(LAD_DBF)
    segments = read_shp_polygon_segments(LAD_SHP)
    out = {}
    for rec, segs in zip(records, segments):
        city = rec.get('LAD23NM')
        if city in CITY_ORDER:
            city_segments = []
            for seg in segs:
                converted = []
                for east, north in seg:
                    lon, lat = osgb36_to_lonlat(east, north)
                    converted.append(relative_xy(city, lon, lat))
                city_segments.append(converted)
            out[city] = city_segments
    return out


LAD_BOUNDARIES = load_lad_boundaries()


def city_extent(city):
    pts = ASSIGN.loc[ASSIGN.city.eq(city), ['x_m','y_m']].dropna()
    xmin, xmax = pts.x_m.min(), pts.x_m.max()
    ymin, ymax = pts.y_m.min(), pts.y_m.max()
    pad_x = (xmax - xmin) * 0.06
    pad_y = (ymax - ymin) * 0.06
    return xmin-pad_x, xmax+pad_x, ymin-pad_y, ymax+pad_y


def city_extent_with_boundary(city):
    xmin, xmax, ymin, ymax = city_extent(city)
    segs = LAD_BOUNDARIES.get(city, [])
    if segs:
        xs = [x for seg in segs for x, _ in seg]
        ys = [y for seg in segs for _, y in seg]
        xmin = min(xmin, min(xs))
        xmax = max(xmax, max(xs))
        ymin = min(ymin, min(ys))
        ymax = max(ymax, max(ys))
        pad_x = (xmax - xmin) * 0.035
        pad_y = (ymax - ymin) * 0.035
        xmin, xmax, ymin, ymax = xmin-pad_x, xmax+pad_x, ymin-pad_y, ymax+pad_y
    return xmin, xmax, ymin, ymax


def make_transform(extent, rect, fixed_scale=None, align_x=0.5):
    xmin, xmax, ymin, ymax = extent
    x0, y0, w, h = rect
    sx = w / (xmax - xmin)
    sy = h / (ymax - ymin)
    s = fixed_scale if fixed_scale is not None else min(sx, sy)
    used_w = (xmax - xmin) * s
    used_h = (ymax - ymin) * s
    ox = x0 + (w - used_w) * align_x
    oy = y0 + (h - used_h) / 2
    def tr(x, y):
        return ox + (x - xmin) * s, oy + used_h - (y - ymin) * s
    return tr, s, (ox, oy, used_w, used_h)


def circle(draw, x, y, r, fill, outline=None, width=2):
    draw.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline=outline or fill, width=width)


def draw_north(draw, x, y, size=42):
    pts = [(x, y-size), (x-size*0.35, y+size*0.45), (x, y+size*0.12), (x+size*0.35, y+size*0.45)]
    draw.polygon(pts, fill='white', outline=PAL['dark'])
    inner = [(x, y-size*0.70), (x-size*0.10, y+size*0.05), (x, y-size*0.10), (x+size*0.10, y+size*0.05)]
    draw.polygon(inner, fill=PAL['dark'])
    text(draw, (x, y+size*0.65), 'N', F_TINY, PAL['dark'], anchor='mm')


def draw_scale(draw, transform, y_m, x_m, length=5000, label='5 km'):
    x1, y1 = transform(x_m, y_m)
    x2, y2 = transform(x_m + length, y_m)
    draw.line([(x1, y1), (x2, y2)], fill=PAL['dark'], width=4)
    for xx in [x1, (x1+x2)/2, x2]:
        draw.line([(xx, y1-8), (xx, y1+8)], fill=PAL['dark'], width=2)
    text(draw, (x1, y1+16), '0', F_TINY, PAL['dark'], anchor='ma')
    text(draw, ((x1+x2)/2, y1+16), '2.5', F_TINY, PAL['dark'], anchor='ma')
    text(draw, (x2, y1+16), label, F_TINY, PAL['dark'], anchor='ma')


def draw_scale_left(draw, rect, scale, length=5000, label='5 km'):
    x0, y0, w, h = rect
    x1 = x0 + 58
    y1 = y0 + h - 42
    x2 = x1 + length * scale
    draw.line([(x1, y1), (x2, y1)], fill=PAL['dark'], width=4)
    for xx in [x1, (x1 + x2) / 2, x2]:
        draw.line([(xx, y1 - 8), (xx, y1 + 8)], fill=PAL['dark'], width=2)
    text(draw, (x1, y1 + 16), '0', F_TINY, PAL['dark'], anchor='ma')
    mid_label = f'{length / 2000:g}'
    end_label = f'{length / 1000:g} km'
    text(draw, ((x1 + x2) / 2, y1 + 16), mid_label, F_TINY, PAL['dark'], anchor='ma')
    text(draw, (x2, y1 + 16), end_label, F_TINY, PAL['dark'], anchor='ma')


def draw_cluster_panel_legend(draw, rect, win):
    x0, y0, w, h = rect
    x = x0 + w - 300
    y = y0 + h - 145
    fill = PAL['early'] if win == 2016 else PAL['late']
    circle(draw, x, y, 13, fill, outline='white')
    text(draw, (x + 28, y - 11), f'{win} cluster centroid', F_TINY, PAL['ink'])
    y += 42
    circle(draw, x + 2, y, cluster_radius(500) * 0.72, '#DCE8EF', outline=PAL['line'])
    circle(draw, x + 54, y, cluster_radius(1500) * 0.72, '#DCE8EF', outline=PAL['line'])
    text(draw, (x + 105, y - 11), 'Circle size = records', F_TINY, PAL['ink'])
    y += 48
    draw.line([(x - 2, y), (x + 55, y)], fill=PAL['boundary'], width=3)
    text(draw, (x + 72, y - 11), 'Local authority boundary', F_TINY, PAL['ink'])


def draw_panel_frame(draw, rect):
    x, y, w, h = rect
    draw.rectangle((x, y, x+w, y+h), outline=PAL['dark'], width=3, fill=PAL['bg'])


def draw_boundary_outline(draw, city, transform):
    for seg in LAD_BOUNDARIES.get(city, []):
        pts = [transform(x, y) for x, y in seg]
        if len(pts) > 1:
            draw.line(pts, fill=PAL['boundary'], width=3)


def draw_centroid_shift_legend(draw, rect):
    x0, y0, w, h = rect
    lx = x0 + w - 405
    ly = y0 + h - 150
    circle(draw, lx, ly, 10, 'white', outline=PAL['line'], width=3)
    text(draw, (lx + 30, ly - 14), '2016 centroid', F_SMALL, PAL['ink'])
    circle(draw, lx, ly + 35, 10, PAL['line'], outline='white', width=3)
    text(draw, (lx + 30, ly + 21), '2025 centroid', F_SMALL, PAL['ink'])
    draw.line([(lx - 10, ly + 72), (lx + 82, ly + 72)], fill=PAL['accent'], width=8)
    text(draw, (lx + 104, ly + 58), 'Selected largest centroid shifts', F_SMALL, PAL['ink'])
    draw.line([(lx - 10, ly + 110), (lx + 82, ly + 110)], fill=PAL['boundary'], width=4)
    text(draw, (lx + 104, ly + 96), 'Local authority boundary', F_SMALL, PAL['ink'])


def draw_centroid_shift_legend_large(draw, rect, median_d, max_d):
    x0, y0, w, h = rect
    lx = x0 + w - 620
    ly = y0 + h - 520
    text(draw, (lx, ly), f'Median shift: {median_d:,.0f} m | Maximum: {max_d:,.0f} m', F_HALF_NOTE, PAL['ink'])
    circle(draw, lx + 12, ly + 67, 12, 'white', outline=PAL['line'], width=3)
    text(draw, (lx + 42, ly + 45), '2016 centroid', F_HALF_LEG, PAL['ink'])
    circle(draw, lx + 12, ly + 115, 12, PAL['line'], outline='white', width=3)
    text(draw, (lx + 42, ly + 93), '2025 centroid', F_HALF_LEG, PAL['ink'])
    draw.line([(lx - 2, ly + 174), (lx + 120, ly + 174)], fill=PAL['accent'], width=8)
    text(draw, (lx + 150, ly + 151), 'Selected largest centroid shifts', F_HALF_LEG, PAL['ink'])
    draw.line([(lx - 2, ly + 232), (lx + 120, ly + 232)], fill=PAL['boundary'], width=4)
    text(draw, (lx + 150, ly + 209), 'Local authority boundary', F_HALF_LEG, PAL['ink'])


def cluster_radius(n):
    return 6 + math.sqrt(max(n, 1)) * 0.65


def fig_414():
    img = Image.new('RGB', (2100, 1118), PAL['bg'])
    draw = ImageDraw.Draw(img)
    rects = {
        ('Birmingham', 2016): (35, 55, 1000, 495),
        ('Birmingham', 2025): (1065, 55, 1000, 495),
        ('Liverpool', 2016): (35, 610, 1000, 495),
        ('Liverpool', 2025): (1065, 610, 1000, 495),
    }
    extents = {city: city_extent_with_boundary(city) for city in CITY_ORDER}
    common_scale = None
    for city in CITY_ORDER:
        xmin, xmax, ymin, ymax = extents[city]
        for win in [2016, 2025]:
            rect = rects[(city, win)]
            city_scale = min(rect[2] / (xmax - xmin), rect[3] / (ymax - ymin))
            common_scale = city_scale if common_scale is None else min(common_scale, city_scale)
    for city in CITY_ORDER:
        for win in [2016, 2025]:
            rect = rects[(city, win)]
            draw.rectangle((rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]), fill='#FFFFFF', outline='#111111', width=2)
            tr, scale, used = make_transform(extents[city], rect, fixed_scale=common_scale)
            text(draw, (rect[0], rect[1]-36), f'{city}, {win}', F_PANEL, PAL['dark'])
            draw_boundary_outline(draw, city, tr)
            pts = ASSIGN[(ASSIGN.city.eq(city)) & (ASSIGN.window.eq(win)) & (ASSIGN.cluster_id.ne(-1))]
            step = max(1, len(pts)//4500)
            for _, row in pts.iloc[::step].iterrows():
                px, py = tr(row.x_m, row.y_m)
                draw.point((px, py), fill=PAL['point'])
            ss = SUMMARY[(SUMMARY.city.eq(city)) & (SUMMARY.window.eq(win))].sort_values('n_records')
            for _, row in ss.iterrows():
                px, py = tr(row.centroid_x_m, row.centroid_y_m)
                r = cluster_radius(row.n_records)
                fill = PAL['early'] if win == 2016 else PAL['late']
                circle(draw, px, py, r, fill=fill, outline='white', width=3)
            draw_north(draw, rect[0]+58, rect[1]+84, 38)
            draw_scale_left(draw, rect, scale, length=5000)
            draw_cluster_panel_legend(draw, rect, win)
    save_all(img, 'figure_4_14_cluster_distribution_2016_vs_2025')


def fig_415():
    counts = MOVES.groupby(['city','status']).size().reset_index(name='n')
    counts.to_csv(OUT / 'figure_4_15_persistence_emergence_counts.csv', index=False, encoding='utf-8-sig')
    img = Image.new('RGB', (2200, 1280), PAL['bg'])
    draw = ImageDraw.Draw(img)
    title_block(draw, '4.15', 'Cluster Persistence and Emergence',
                'Status classification compares 2016 and 2025 HDBSCAN cluster centroids.')
    plot = (240, 105, 1750, 880)
    x0, y0, w, h = plot
    maxv = max(30, int(counts.n.max()) + 4)
    for val in range(0, maxv+1, 5):
        yy = y0 + h - val/maxv*h
        draw.line([(x0, yy), (x0+w, yy)], fill=PAL['grid'], width=3)
        text(draw, (x0-24, yy-13), str(val), font(30), PAL['muted'], anchor='ra')
    draw.line([(x0, y0), (x0, y0+h), (x0+w, y0+h)], fill=PAL['line'], width=4)
    statuses = ['persistent_or_relocated','emerged','disappeared']
    group_centres = [x0 + w*0.28, x0 + w*0.72]
    bw = 150
    city_status_counts = {
        city: {
            st: int(counts[(counts.city.eq(city)) & (counts.status.eq(st))].n.sum())
            for st in statuses
        }
        for city in CITY_ORDER
    }
    for gi, city in enumerate(CITY_ORDER):
        cx = group_centres[gi]
        text(draw, (cx, y0+h+62), city, font(42, bold=True), PAL['dark'], anchor='mm')
        denom = city_status_counts[city]['persistent_or_relocated'] + city_status_counts[city]['disappeared']
        for si, st in enumerate(statuses):
            val = city_status_counts[city][st]
            bx = cx + (si-1)*215 - bw/2
            if val > 0:
                by = y0 + h - val/maxv*h
                draw.rounded_rectangle((bx, by, bx+bw, y0+h), radius=12, fill=STATUS_COLOR[st])
            else:
                by = y0 + h
            if st == 'persistent_or_relocated' and denom:
                label = f"{val} ({val / denom * 100:.1f}%)"
            else:
                label = str(val)
            text(draw, (bx+bw/2, by-42), label, font(32, bold=True), PAL['dark'], anchor='mm')
    rotated_text(img, (x0 - 140, y0 + h / 2), 'Number of clusters', font(36), PAL['ink'], angle=90)
    # Legend
    lx, ly = 560, 1190
    for st in statuses:
        draw.rounded_rectangle((lx, ly, lx+46, ly+30), radius=7, fill=STATUS_COLOR[st])
        text(draw, (lx+64, ly-1), STATUS_LABEL[st], font(28), PAL['ink'])
        lx += 505
    source_note(draw, 1225)
    save_all(img, 'figure_4_15_persistence_emergence')


def fig_416():
    img = Image.new('RGB', (1740, 2540), PAL['bg'])
    draw = ImageDraw.Draw(img)
    title_block(draw, '4.16', 'Centroid Shift of Persistent / Repositioned Clusters',
                'Arrows connect matched 2016 and 2025 cluster centroids; labels mark the largest shifts.')
    rects = {'Birmingham': (60, 90, 1620, 1120), 'Liverpool': (60, 1345, 1620, 1120)}
    pers = MOVES[MOVES.status.eq('persistent_or_relocated')].copy()
    pers.to_csv(OUT / 'figure_4_16_centroid_shift_vectors.csv', index=False, encoding='utf-8-sig')
    extents = {city: city_extent_with_boundary(city) for city in CITY_ORDER}
    common_scale = None
    for city in CITY_ORDER:
        xmin, xmax, ymin, ymax = extents[city]
        _, _, rw, rh = rects[city]
        city_scale = min(rw / (xmax - xmin), rh / (ymax - ymin))
        common_scale = city_scale if common_scale is None else min(common_scale, city_scale)
    for city in CITY_ORDER:
        rect = rects[city]
        draw_panel_frame(draw, rect)
        extent = extents[city]
        xmin, xmax, ymin, ymax = extent
        tr, scale, used = make_transform(extent, rect, fixed_scale=common_scale)
        text(draw, (rect[0], rect[1]-58), city, F_PANEL, PAL['dark'])
        draw_boundary_outline(draw, city, tr)
        pts = ASSIGN[(ASSIGN.city.eq(city)) & (ASSIGN.window.isin([2016,2025])) & (ASSIGN.cluster_id.ne(-1))]
        step = max(1, len(pts)//5000)
        for _, row in pts.iloc[::step].iterrows():
            px, py = tr(row.x_m, row.y_m)
            draw.point((px, py), fill='#C4CED6')
        city_p = pers[pers.city.eq(city)].sort_values('distance_m', ascending=False)
        top_ids = set(city_p.head(3).index)
        for idx, row in city_p.sort_values('distance_m').iterrows():
            x1, y1 = tr(row.early_centroid_x_m, row.early_centroid_y_m)
            x2, y2 = tr(row.late_centroid_x_m, row.late_centroid_y_m)
            col = PAL['accent'] if idx in top_ids else PAL['line']
            width = 8 if idx in top_ids else 3
            draw.line([(x1,y1),(x2,y2)], fill=col, width=width)
            ang = math.atan2(y2-y1, x2-x1)
            ah = 24 if idx in top_ids else 13
            pts_arrow = [(x2,y2), (x2-ah*math.cos(ang-0.45), y2-ah*math.sin(ang-0.45)),
                         (x2-ah*math.cos(ang+0.45), y2-ah*math.sin(ang+0.45))]
            draw.polygon(pts_arrow, fill=col)
            circle(draw, x1, y1, 8 if idx in top_ids else 6, fill='white', outline=col, width=3 if idx in top_ids else 2)
            circle(draw, x2, y2, 8 if idx in top_ids else 6, fill=col, outline='white', width=3 if idx in top_ids else 2)
        for _, row in city_p.head(3).iterrows():
            x2, y2 = tr(row.late_centroid_x_m, row.late_centroid_y_m)
            dist = int(round(row.distance_m))
            label = f'{dist} m'
            dx, dy = 18, -18
            if city == 'Liverpool':
                if dist < 700:
                    dx, dy = 30, -34
                elif dist < 1060:
                    dx, dy = 30, 18
                else:
                    dx, dy = 24, -8
            text(draw, (x2+dx, y2+dy), label, F_DISTANCE, PAL['dark'])
        draw_north(draw, rect[0]+58, rect[1]+82, 38)
        draw_scale_left(draw, rect, scale, length=3000 if city == 'Liverpool' else 5000)
        median_d = city_p.distance_m.median()
        max_d = city_p.distance_m.max()
        text(draw, (rect[0]+rect[2]-42, rect[1]+rect[3]-190), f'Median shift: {median_d:,.0f} m | Maximum: {max_d:,.0f} m', F_SMALL, PAL['ink'], anchor='ra')
        draw_centroid_shift_legend(draw, rect)
    save_all(img, 'figure_4_16_centroid_shift')


def fig_416_single_maps():
    pers = MOVES[MOVES.status.eq('persistent_or_relocated')].copy()
    extents = {city: city_extent_with_boundary(city) for city in CITY_ORDER}
    rect = (75, 120, 2260, 1400)
    canvas = (2500, 1700)
    _, birmingham_single_scale, _ = make_transform(extents['Birmingham'], rect, fixed_scale=None, align_x=0.31)
    for city in CITY_ORDER:
        img = Image.new('RGB', canvas, PAL['bg'])
        draw = ImageDraw.Draw(img)
        draw.rectangle(
            (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]),
            fill=PAL['bg'],
            outline=PAL['dark'],
            width=3,
        )
        text(draw, (rect[0] + rect[2] / 2, rect[1] - 62), city, F_HALF_TITLE, PAL['dark'], anchor='mm')
        fixed_single_scale = birmingham_single_scale if city == 'Liverpool' else None
        tr, scale, _ = make_transform(extents[city], rect, fixed_scale=fixed_single_scale, align_x=0.31)

        pts = ASSIGN[(ASSIGN.city.eq(city)) & (ASSIGN.window.isin([2016, 2025])) & (ASSIGN.cluster_id.ne(-1))]
        step = max(1, len(pts) // 5000)
        for _, row in pts.iloc[::step].iterrows():
            px, py = tr(row.x_m, row.y_m)
            draw.point((px, py), fill='#D8DEE4')

        draw_boundary_outline(draw, city, tr)

        city_p = pers[pers.city.eq(city)].sort_values('distance_m', ascending=False)
        top_ids = set(city_p.head(3).index)
        for idx, row in city_p.sort_values('distance_m').iterrows():
            x1, y1 = tr(row.early_centroid_x_m, row.early_centroid_y_m)
            x2, y2 = tr(row.late_centroid_x_m, row.late_centroid_y_m)
            col = PAL['accent'] if idx in top_ids else PAL['line']
            width = 10 if idx in top_ids else 4
            draw.line([(x1, y1), (x2, y2)], fill=col, width=width)
            ang = math.atan2(y2 - y1, x2 - x1)
            ah = 30 if idx in top_ids else 16
            arrow_pts = [
                (x2, y2),
                (x2 - ah * math.cos(ang - 0.45), y2 - ah * math.sin(ang - 0.45)),
                (x2 - ah * math.cos(ang + 0.45), y2 - ah * math.sin(ang + 0.45)),
            ]
            draw.polygon(arrow_pts, fill=col)
            circle(draw, x1, y1, 10 if idx in top_ids else 7, fill='white', outline=col, width=3)
            circle(draw, x2, y2, 10 if idx in top_ids else 7, fill=col, outline='white', width=3)

        for _, row in city_p.head(3).iterrows():
            x2, y2 = tr(row.late_centroid_x_m, row.late_centroid_y_m)
            dist = int(round(row.distance_m))
            dx, dy = 24, -26
            if city == 'Liverpool':
                if dist < 700:
                    dx, dy = 42, -48
                elif dist < 1060:
                    dx, dy = 38, 30
                else:
                    dx, dy = 34, -10
            text(draw, (x2 + dx, y2 + dy), f'{dist} m', F_HALF_DISTANCE, PAL['dark'])

        draw_north(draw, rect[0] + 105, rect[1] + 150, 66)
        draw_scale_left(draw, rect, scale, length=5000)
        draw_centroid_shift_legend_large(draw, rect, city_p.distance_m.median(), city_p.distance_m.max())
        save_all(img, f'figure_4_16_centroid_shift_{city.lower()}_halfpage')


def fig_417():
    late = SUMMARY[SUMMARY.window.eq(2025)].copy()
    mov_status = MOVES[['city','late_cluster_id','status']].dropna().copy()
    mov_status['late_cluster_id'] = mov_status['late_cluster_id'].astype(int)
    late = late.merge(mov_status, how='left', left_on=['city','cluster_id'], right_on=['city','late_cluster_id'])
    late['status'] = late['status'].fillna('unmatched')
    rows = []
    for city in CITY_ORDER:
        top = late[late.city.eq(city)].sort_values('n_records', ascending=False).head(12).copy()
        top['rank_2025'] = range(1, len(top)+1)
        rows.append(top)
    rank = pd.concat(rows, ignore_index=True)
    rank.to_csv(OUT / 'figure_4_17_cluster_ranking_2025_top12.csv', index=False, encoding='utf-8-sig')
    concentration_note = {
        'Birmingham': ('Largest cluster: 12.3%', 'Top 3 clusters: 28.7%'),
        'Liverpool': ('Largest cluster: 27.3%', 'Top 3 clusters: 49.6%'),
    }
    img = Image.new('RGB', (2500, 1450), PAL['bg'])
    draw = ImageDraw.Draw(img)
    title_block(draw, '4.17', 'Cluster Ranking by 2025 Retail Records',
                'Top 2025 HDBSCAN clusters ranked by cluster size; colour indicates cluster status.')
    rects = {'Birmingham': (180, 105, 980, 1000), 'Liverpool': (1320, 105, 980, 1000)}
    maxv = int(rank.n_records.max() * 1.18)
    for city in CITY_ORDER:
        rect = rects[city]
        x0, y0, w, h = rect
        text(draw, (x0, y0-50), city, font(42, bold=True), PAL['dark'])
        note_x = x0 + w - 430
        note_y = y0 + 110
        text(draw, (note_x, note_y), concentration_note[city][0], font(29, bold=True), PAL['dark'])
        text(draw, (note_x, note_y + 46), concentration_note[city][1], font(29, bold=True), PAL['dark'])
        city_rank = rank[rank.city.eq(city)].sort_values('rank_2025', ascending=False)
        n = len(city_rank)
        row_h = h / (n + 1)
        axis_x = x0 + 205
        bar_max = w - 285
        for gv in [0, 500, 1000, 1500, 2000]:
            if gv <= maxv:
                xx = axis_x + gv/maxv*bar_max
                draw.line([(xx, y0+32), (xx, y0+h-40)], fill=PAL['grid'], width=2)
                text(draw, (xx, y0+h+24), str(gv), font(24), PAL['muted'], anchor='ma')
        for i, (_, row) in enumerate(city_rank.iterrows()):
            yy = y0 + 68 + i*row_h
            label = f'#{int(row.rank_2025)}  C{int(row.cluster_id)}'
            text(draw, (x0+20, yy-13), label, font(28), PAL['ink'])
            bw = row.n_records / maxv * bar_max
            st = row.status if row.status in STATUS_COLOR else 'persistent_or_relocated'
            col = STATUS_COLOR.get(st, PAL['neutral'])
            draw.rounded_rectangle((axis_x, yy-23, axis_x+bw, yy+23), radius=10, fill=col)
            text(draw, (axis_x+bw+18, yy-15), f'{int(row.n_records):,}', font(28), PAL['ink'])
        draw.line([(axis_x, y0+h-40), (axis_x+bar_max, y0+h-40)], fill=PAL['line'], width=3)
        text(draw, (axis_x+bar_max/2, y0+h+76), 'Retail records in cluster, 2025', font(28), PAL['muted'], anchor='mm')
    lx, ly = 885, 1320
    for st in ['persistent_or_relocated','emerged']:
        draw.rounded_rectangle((lx, ly, lx+48, ly+30), radius=7, fill=STATUS_COLOR[st])
        text(draw, (lx+66, ly-1), STATUS_LABEL[st], font(29), PAL['ink'])
        lx += 520
    source_note(draw, 1405)
    save_all(img, 'figure_4_17_cluster_ranking')


if __name__ == '__main__':
    fig_414()
    fig_415()
    fig_416()
    fig_416_single_maps()
    fig_417()
    print('Wrote cluster figures to', OUT)



