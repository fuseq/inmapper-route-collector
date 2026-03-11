"""
Expocad JSON → inMapper SVG Converter

Expocad webdrawing API'sinden alınan JSON verisini,
inMapper'ın fuar SVG formatına dönüştürür.

Kullanım:
    py tools/expocad_to_svg.py <expocad_url_or_json_file> -o output.svg [options]

Örnekler:
    py tools/expocad_to_svg.py https://gsma.expocad.com/25mwcb -o mwc_floor.svg
    py tools/expocad_to_svg.py tools/expocad_webdrawing.json -o mwc_floor.svg --width 800
"""

import json
import math
import argparse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional


# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class Point:
    x: float
    y: float

@dataclass
class BoothGeometry:
    corners: list[Point]
    text_start: Optional[Point] = None

@dataclass
class Booth:
    index: int
    number: str
    name: str
    status: int
    color: str
    outline_color: str
    area_sqm: str
    geometries: list[BoothGeometry] = field(default_factory=list)
    text_items: list[dict] = field(default_factory=list)

@dataclass
class Pavilion:
    name: str
    color: str
    geometries: list[BoothGeometry] = field(default_factory=list)
    text_items: list[dict] = field(default_factory=list)

@dataclass
class GraphicShape:
    color: str
    width: str
    filled: bool
    geometries: list[BoothGeometry] = field(default_factory=list)

@dataclass
class TextLabel:
    x: float
    y: float
    text: str
    rotation: float
    size: float
    color: Optional[str] = None

@dataclass
class ExpocadData:
    event_name: str
    extent: Point
    booths: list[Booth] = field(default_factory=list)
    pavilions: list[Pavilion] = field(default_factory=list)
    building_graphics: list[GraphicShape] = field(default_factory=list)
    building_texts: list[TextLabel] = field(default_factory=list)
    tech_layers: list[dict] = field(default_factory=list)
    config: Optional[dict] = None


# ── JSON Parser ──────────────────────────────────────────────────────────────

def parse_geometry(geom_list: list[dict]) -> list[BoothGeometry]:
    result = []
    for g in geom_list:
        corners = [Point(c["x"], c["y"]) for c in g.get("corners", [])]
        tsp = g.get("textStartPoint")
        text_start = Point(tsp["x"], tsp["y"]) if tsp else None
        if corners:
            result.append(BoothGeometry(corners=corners, text_start=text_start))
    return result


def parse_expocad_json(webdrawing: list[dict], config: list[dict]) -> ExpocadData:
    entry = webdrawing[0]
    cfg = config[0] if config else {}

    extent = Point(
        entry.get("Extent", {}).get("x", 17280),
        entry.get("Extent", {}).get("y", 11112),
    )

    data = ExpocadData(
        event_name=entry.get("eventName", "unknown"),
        extent=extent,
        config=cfg,
    )

    for b in entry.get("booths", []):
        booth = Booth(
            index=int(b.get("boothIndex", 0)),
            number=b.get("boothNumber", ""),
            name=b.get("displayName", "").strip("* ").strip(),
            status=int(b.get("status", 0)),
            color=b.get("shadeColor", b.get("colorOverride", "#d9d3d2")),
            outline_color=b.get("outlineColor", "#000000"),
            area_sqm=b.get("areaSqM", ""),
            geometries=parse_geometry(b.get("geometry", [])),
            text_items=b.get("textItems", []),
        )
        if booth.geometries:
            data.booths.append(booth)

    for p in entry.get("pavilions", []):
        pav = Pavilion(
            name=p.get("name", ""),
            color=p.get("colorOverride", "#e6e6e6"),
            geometries=parse_geometry(p.get("geometry", [])),
            text_items=p.get("textItems", []),
        )
        if pav.geometries:
            data.pavilions.append(pav)

    ag = entry.get("additionalGraphics", {})
    if isinstance(ag, dict):
        for g in ag.get("graphics", []):
            shape = GraphicShape(
                color=g.get("colorOverride", "#000000"),
                width=g.get("w", "1"),
                filled=g.get("b", "F") == "T",
                geometries=parse_geometry(g.get("geometry", [])),
            )
            if shape.geometries:
                data.building_graphics.append(shape)

        for t in ag.get("texts", []):
            data.building_texts.append(TextLabel(
                x=float(t.get("X", 0)),
                y=float(t.get("Y", 0)),
                text=t.get("text", ""),
                rotation=float(t.get("rotationRads", 0)),
                size=float(t.get("size", 10)),
                color=t.get("color"),
            ))

    return data


# ── Coordinate Transform ─────────────────────────────────────────────────────

class CoordinateTransform:
    def __init__(self, extent: Point, target_width: int = 800, padding: int = 10):
        self.extent = extent
        self.padding = padding

        aspect = extent.y / extent.x if extent.x else 1
        self.svg_width = target_width
        self.svg_height = int(target_width * aspect)

        self.scale_x = (target_width - 2 * padding) / extent.x
        self.scale_y = (self.svg_height - 2 * padding) / extent.y
        self.scale = min(self.scale_x, self.scale_y)

    def transform(self, p: Point) -> Point:
        return Point(
            round(p.x * self.scale + self.padding, 2),
            round(p.y * self.scale + self.padding, 2),
        )

    def stroke_width(self, reference_sw: float, reference_viewport: int = 7271) -> str:
        """Scale stroke-width proportionally to match fuar SVG appearance."""
        return str(round(reference_sw * self.svg_width / reference_viewport, 2))


# ── SVG Path Generator ───────────────────────────────────────────────────────

PATH_STYLE = "font-variation-settings:normal;-inkscape-stroke:none"


def corners_to_svg_path(corners: list[Point], transform: CoordinateTransform) -> str:
    if not corners:
        return ""

    parts = []
    first = transform.transform(corners[0])
    parts.append(f"m {first.x},{first.y}")

    prev = first
    for pt in corners[1:]:
        tp = transform.transform(pt)
        dx = round(tp.x - prev.x, 2)
        dy = round(tp.y - prev.y, 2)
        parts.append(f"{dx},{dy}")
        prev = tp

    parts.append("z")
    return " ".join(parts)


def centroid(corners: list[Point]) -> Point:
    if not corners:
        return Point(0, 0)
    cx = sum(p.x for p in corners) / len(corners)
    cy = sum(p.y for p in corners) / len(corners)
    return Point(cx, cy)


# ── SVG Builder ──────────────────────────────────────────────────────────────

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
SODIPODI_NS = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"


def _layer(parent, layer_id, label=None, **extra_attrs):
    attrs = {
        f"{{{INKSCAPE_NS}}}groupmode": "layer",
        "id": layer_id,
        f"{{{INKSCAPE_NS}}}label": label or layer_id,
    }
    attrs.update(extra_attrs)
    return ET.SubElement(parent, "g", attrs)


def build_svg(data: ExpocadData, target_width: int = 800) -> ET.Element:
    tx = CoordinateTransform(data.extent, target_width)
    sw = lambda ref: tx.stroke_width(ref)

    ET.register_namespace("", SVG_NS)
    ET.register_namespace("inkscape", INKSCAPE_NS)
    ET.register_namespace("sodipodi", SODIPODI_NS)

    svg = ET.Element(f"{{{SVG_NS}}}svg", {
        "version": "1.1",
        "id": "svg_expocad",
        "width": str(tx.svg_width),
        "height": str(tx.svg_height),
        "viewBox": f"0 0 {tx.svg_width} {tx.svg_height}",
        f"{{{SODIPODI_NS}}}docname": "0.svg",
        f"{{{INKSCAPE_NS}}}version": "1.1 (c4e8f9e, 2021-05-24)",
    })

    # ── namedview ─────────────────────────────────────────────────────────
    ET.SubElement(svg, f"{{{SODIPODI_NS}}}namedview", {
        "id": "namedview826",
        "pagecolor": "#ffffff",
        "bordercolor": "#666666",
        "borderopacity": "1",
        f"{{{INKSCAPE_NS}}}pageshadow": "2",
        f"{{{INKSCAPE_NS}}}pageopacity": "0",
        f"{{{INKSCAPE_NS}}}pagecheckerboard": "0",
        "showgrid": "false",
        f"{{{INKSCAPE_NS}}}current-layer": "Writing",
        f"{{{INKSCAPE_NS}}}object-nodes": "true",
        f"{{{INKSCAPE_NS}}}object-paths": "false",
    })

    # ── Rooms ─────────────────────────────────────────────────────────────

    rooms = _layer(svg, "Rooms", display="inline", **{
        "fill-opacity": "1",
        "stroke": "#969696",
        "stroke-opacity": "1",
    })

    # Walking
    walking_grp = _layer(rooms, "Walking", display="inline")

    all_xs, all_ys = [], []
    for booth in data.booths:
        for geom in booth.geometries:
            for c in geom.corners:
                all_xs.append(c.x)
                all_ys.append(c.y)
    for pav in data.pavilions:
        for geom in pav.geometries:
            for c in geom.corners:
                all_xs.append(c.x)
                all_ys.append(c.y)

    if all_xs and all_ys:
        margin = 80
        min_pt = tx.transform(Point(min(all_xs) - margin, min(all_ys) - margin))
        max_pt = tx.transform(Point(max(all_xs) + margin, max(all_ys) + margin))
        walking_d = (
            f"m {min_pt.x},{min_pt.y} "
            f"{round(max_pt.x - min_pt.x, 2)},0 "
            f"0,{round(max_pt.y - min_pt.y, 2)} "
            f"{round(min_pt.x - max_pt.x, 2)},0 z"
        )
        ET.SubElement(walking_grp, "path", {
            "id": "walking_area",
            "d": walking_d,
            "fill": "#f5f5f5",
            "stroke-width": sw(8),
            "style": PATH_STYLE,
        })

    # Building
    building_grp = _layer(rooms, "Building", display="inline", **{
        "fill": "#e6e6e6",
        "stroke-width": sw(2),
    })

    for i, pav in enumerate(data.pavilions):
        for j, geom in enumerate(pav.geometries):
            d = corners_to_svg_path(geom.corners, tx)
            if d:
                ET.SubElement(building_grp, "path", {
                    "id": f"pav{i}_{j}",
                    "d": d,
                    "style": PATH_STYLE,
                })

    for i, shape in enumerate(data.building_graphics):
        for j, geom in enumerate(shape.geometries):
            d = corners_to_svg_path(geom.corners, tx)
            if d:
                ET.SubElement(building_grp, "path", {
                    "id": f"bg{i}_{j}",
                    "d": d,
                    "style": PATH_STYLE,
                })

    # Water (empty)
    _layer(rooms, "Water", display="inline", **{
        "fill": "#cfe2f3",
        "stroke-width": sw(2),
    })

    # Service
    service_grp = _layer(rooms, "Service", display="inline", **{
        "fill": "#e9dad0",
        "stroke-width": sw(2),
    })

    # Food
    food_grp = _layer(rooms, "Food", display="inline", **{
        "fill": "#d1bbbc",
        "stroke-width": sw(2),
    })

    # Stand (booths)
    stand_grp = _layer(rooms, "Stand", display="inline", **{
        "fill": "#d9d3d2",
        "stroke-width": sw(2),
    })

    booth_id_counter = 1
    booth_id_map = {}

    for booth in data.booths:
        if booth.status == 2:
            target_grp = stand_grp
        elif booth.status == 1:
            target_grp = service_grp
        else:
            target_grp = service_grp

        geom_count = len(booth.geometries)
        base_id = f"ID{booth_id_counter:03d}"

        for gi, geom in enumerate(booth.geometries):
            d = corners_to_svg_path(geom.corners, tx)
            if d:
                if geom_count > 1:
                    path_id = f"{base_id}_{gi + 1}"
                else:
                    path_id = base_id

                ET.SubElement(target_grp, "path", {
                    "id": path_id,
                    "d": d,
                    "style": PATH_STYLE,
                    "data-booth-number": booth.number,
                    "data-booth-name": booth.name,
                })

        booth_id_map[booth_id_counter] = booth
        booth_id_counter += 1

    # ── Paths (empty) ─────────────────────────────────────────────────────

    _layer(svg, "Paths", display="none", **{
        "fill": "none",
        "fill-opacity": "1",
        "stroke": "#008000",
        "stroke-width": sw(6),
        "stroke-opacity": "1",
    })

    # ── Doors (empty) ─────────────────────────────────────────────────────

    _layer(svg, "Doors", display="none", **{
        "fill": "none",
        "fill-opacity": "1",
        "stroke": "#ff0000",
        "stroke-width": sw(6),
        "stroke-opacity": "1",
    })

    # ── Portals (empty) ───────────────────────────────────────────────────

    _layer(svg, "Portals", display="none", **{
        "fill": "none",
        "fill-opacity": "1",
        "stroke": "#0000ff",
        "stroke-width": sw(6),
        "stroke-opacity": "1",
    })

    # ── Writing (booth ID labels - 3-line tspan like fuar SVGs) ─────────

    writing_grp = _layer(svg, "Writing", display="inline")

    TSPAN_STYLE = "text-align:center"

    # Font-size and line-height in 3 categories, proportional to fuar.svg
    # fuar.svg: ~10px font, 12px line spacing at 7271px viewport
    ref_vp = 7271
    fs_small = round(7 * tx.svg_width / ref_vp, 2)
    fs_medium = round(10 * tx.svg_width / ref_vp, 2)
    fs_large = round(14 * tx.svg_width / ref_vp, 2)

    lh_small = round(9 * tx.svg_width / ref_vp, 2)
    lh_medium = round(12 * tx.svg_width / ref_vp, 2)
    lh_large = round(16 * tx.svg_width / ref_vp, 2)

    # Raw area thresholds (expocad coordinate space)
    area_thresh_med = 40000
    area_thresh_large = 200000

    for bid, booth in booth_id_map.items():
        geom_count = len(booth.geometries)
        base_id = f"ID{bid:03d}"

        for gi, geom in enumerate(booth.geometries):
            if not geom.corners or len(geom.corners) < 3:
                continue

            xs = [p.x for p in geom.corners]
            ys = [p.y for p in geom.corners]
            raw_area = (max(xs) - min(xs)) * (max(ys) - min(ys))

            if raw_area >= area_thresh_large:
                fs, lh = fs_large, lh_large
            elif raw_area >= area_thresh_med:
                fs, lh = fs_medium, lh_medium
            else:
                fs, lh = fs_small, lh_small

            writing_style = (
                f"font-size:{fs}px;line-height:100%;"
                f"-inkscape-font-specification:'Catamaran Bold'"
            )

            # Bounding-box center (not polygon centroid) for precise centering
            bbox_min = Point(min(xs), min(ys))
            bbox_max = Point(max(xs), max(ys))
            bbox_center_raw = Point(
                (bbox_min.x + bbox_max.x) / 2,
                (bbox_min.y + bbox_max.y) / 2,
            )
            tc = tx.transform(bbox_center_raw)
            cx_str = str(round(tc.x, 2))

            total_block_h = lh * 2
            start_y = round(tc.y - total_block_h / 2, 2)

            if geom_count > 1:
                sub_id = f"{base_id}_{gi + 1}"
            else:
                sub_id = base_id

            lines = [f"{sub_id}_1_", f"{sub_id}_2_", f"{sub_id}_3_"]

            text_el = ET.SubElement(writing_grp, "text", {
                "x": cx_str,
                "y": str(start_y),
                "text-anchor": "middle",
                "style": writing_style,
            })

            for li, line_text in enumerate(lines):
                tspan_y = round(start_y + li * lh, 2)
                tspan = ET.SubElement(text_el, "tspan", {
                    "x": cx_str,
                    "y": str(tspan_y),
                    "style": TSPAN_STYLE,
                })
                tspan.text = line_text

    # ── Icons (empty, for wc/atm/entrance icons) ─────────────────────────

    _layer(svg, "Icons", display="inline")

    # ── Constants (empty, for entrance/carpark icons) ─────────────────────

    _layer(svg, "Constants", display="inline")

    return svg


# ── Output ───────────────────────────────────────────────────────────────────

def write_svg(svg_element: ET.Element, output_path: str):
    tree = ET.ElementTree(svg_element)
    ET.indent(tree, space="    ")
    with open(output_path, "wb") as f:
        tree.write(f, encoding="UTF-8", xml_declaration=True)
    print(f"SVG written to: {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert Expocad floorplan data to inMapper fair SVG format"
    )
    parser.add_argument(
        "source",
        help="Expocad event URL or path to saved webdrawing JSON file",
    )
    parser.add_argument("-o", "--output", default="output.svg", help="Output SVG file path")
    parser.add_argument("--width", type=int, default=800, help="Target SVG width in px (default: 800)")
    parser.add_argument("--config-json", help="Path to saved config JSON (only with local JSON)")

    args = parser.parse_args()

    if args.source.startswith("http"):
        url = args.source.replace("/Events/", "/").replace("/index.html", "")
        parts = url.rstrip("/").split("/")
        event_id = parts[-1]
        base = "/".join(parts[:-1])

        wd_url = f"{base}/{event_id}/webdrawing?requestedWidth=17280"
        cfg_url = f"{base}/{event_id}/webdrawing/Config?requestedWidth=17280"

        print(f"Fetching webdrawing: {wd_url}")
        with urllib.request.urlopen(wd_url) as resp:
            webdrawing = json.loads(resp.read().decode("utf-8"))

        print(f"Fetching config: {cfg_url}")
        with urllib.request.urlopen(cfg_url) as resp:
            config = json.loads(resp.read().decode("utf-8"))
    else:
        print(f"Loading webdrawing from: {args.source}")
        with open(args.source, "r", encoding="utf-8") as f:
            webdrawing = json.load(f)

        if args.config_json:
            with open(args.config_json, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = [{}]

    data = parse_expocad_json(webdrawing, config)
    print(f"Event: {data.event_name}")
    print(f"Extent: {data.extent.x} x {data.extent.y}")
    print(f"Booths: {len(data.booths)}")
    print(f"Pavilions: {len(data.pavilions)}")
    print(f"Building graphics: {len(data.building_graphics)}")

    svg = build_svg(data, target_width=args.width)
    write_svg(svg, args.output)

    rented = sum(1 for b in data.booths if b.status == 2)
    available = sum(1 for b in data.booths if b.status == 0)
    hold = sum(1 for b in data.booths if b.status == 1)
    print(f"\nBooth summary: {rented} Stand, {available} available, {hold} on hold (Service)")
    print(f"Layers: Rooms(Walking, Building, Water, Service, Food, Stand) + Paths + Doors + Portals + Writing + Icons + Constants")


if __name__ == "__main__":
    main()
