"""
SVG → GeoJSON Converter for inMapper Maps
==========================================
Converts inMapper SVG floor plans to GeoJSON FeatureCollection format
and generates a MapLibre GL JS viewer for visualization.

Usage:
    py tools/svg_to_geojson.py tools/fuar.svg -o tools/fuar.geojson --viewer
    py tools/svg_to_geojson.py tools/fuar.svg --center-lat 41.0 --center-lng 29.0 --scale 0.03
"""

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

# ── SVG Namespace helpers ────────────────────────────────────────────────────

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
SODIPODI_NS = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd"

NS = {"svg": SVG_NS, "inkscape": INKSCAPE_NS, "sodipodi": SODIPODI_NS}


# ── SVG Path Parsing (reuses extract_xml logic) ─────────────────────────────

def parse_path_d(d: str):
    tokens = re.findall(
        r"[a-df-zA-DF-Z]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", d
    )
    commands = []
    i = 0
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]
            i += 1
            coords = []
            while i < len(tokens) and not tokens[i].isalpha():
                coords.append(float(tokens[i]))
                i += 1
            commands.append((cmd, coords))
        else:
            i += 1
    return commands


def path_to_absolute_coords(d: str) -> List[Tuple[float, float]]:
    """Convert SVG path data to list of absolute (x, y) points."""
    commands = parse_path_d(d)
    points: List[Tuple[float, float]] = []
    cx, cy = 0.0, 0.0
    sx, sy = 0.0, 0.0

    for cmd, coords in commands:
        if cmd == "m":
            if not points:
                cx, cy = coords[0], coords[1]
            else:
                cx += coords[0]
                cy += coords[1]
            sx, sy = cx, cy
            points.append((cx, cy))
            for i in range(2, len(coords), 2):
                cx += coords[i]
                cy += coords[i + 1]
                points.append((cx, cy))
        elif cmd == "M":
            cx, cy = coords[0], coords[1]
            sx, sy = cx, cy
            points.append((cx, cy))
            for i in range(2, len(coords), 2):
                cx, cy = coords[i], coords[i + 1]
                points.append((cx, cy))
        elif cmd == "l":
            for i in range(0, len(coords), 2):
                cx += coords[i]
                cy += coords[i + 1]
                points.append((cx, cy))
        elif cmd == "L":
            for i in range(0, len(coords), 2):
                cx, cy = coords[i], coords[i + 1]
                points.append((cx, cy))
        elif cmd == "h":
            for v in coords:
                cx += v
                points.append((cx, cy))
        elif cmd == "H":
            for v in coords:
                cx = v
                points.append((cx, cy))
        elif cmd == "v":
            for v in coords:
                cy += v
                points.append((cx, cy))
        elif cmd == "V":
            for v in coords:
                cy = v
                points.append((cx, cy))
        elif cmd in ("z", "Z"):
            if points and (cx != sx or cy != sy):
                points.append((sx, sy))
            cx, cy = sx, sy

    return points


def path_to_line_coords(d: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse a simple 2-point SVG path (M + single segment) into (x1,y1,x2,y2)."""
    commands = parse_path_d(d)
    if len(commands) < 2:
        if len(commands) == 1:
            cmd, coords = commands[0]
            if cmd == "m" and len(coords) >= 4:
                x1, y1 = coords[0], coords[1]
                x2 = x1 + coords[2]
                y2 = y1 + coords[3]
                return (x1, y1, x2, y2)
            elif cmd == "M" and len(coords) >= 4:
                return (coords[0], coords[1], coords[2], coords[3])
        return None

    cmd0, c0 = commands[0]
    x1 = c0[0] if c0 else 0.0
    y1 = c0[1] if len(c0) > 1 else 0.0

    cmd1, c1 = commands[1]
    if cmd1 == "h":
        return (x1, y1, x1 + c1[0], y1)
    elif cmd1 == "H":
        return (x1, y1, c1[0], y1)
    elif cmd1 == "v":
        return (x1, y1, x1, y1 + c1[0])
    elif cmd1 == "V":
        return (x1, y1, x1, c1[0])
    elif cmd1 == "l":
        return (x1, y1, x1 + c1[0], y1 + c1[1])
    elif cmd1 == "L":
        return (x1, y1, c1[0], c1[1])
    elif cmd1 == "m":
        return (x1, y1, x1 + c1[0], y1 + c1[1])
    else:
        dx, dy = c1[0], c1[1] if len(c1) > 1 else 0.0
        if cmd0 == "m":
            return (x1, y1, x1 + dx, y1 + dy)
        else:
            return (x1, y1, dx, dy)

    return None


# ── Coordinate Transformation ────────────────────────────────────────────────

class GeoTransform:
    """Maps SVG pixel coordinates to GeoJSON [lng, lat]."""

    def __init__(
        self,
        svg_width: float,
        svg_height: float,
        center_lat: float = 0.0,
        center_lng: float = 0.0,
        scale: float = 0.03,
        origin_x: float = 0.0,
        origin_y: float = 0.0,
    ):
        self.svg_w = svg_width
        self.svg_h = svg_height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.scale = scale

        meters_per_deg_lat = 111_320.0
        meters_per_deg_lng = 111_320.0 * math.cos(math.radians(center_lat))

        self.geo_w = (svg_width * scale) / meters_per_deg_lng
        self.geo_h = (svg_height * scale) / meters_per_deg_lat

        self.min_lng = center_lng - self.geo_w / 2
        self.max_lat = center_lat + self.geo_h / 2

    def to_lnglat(self, x: float, y: float) -> Tuple[float, float]:
        lng = self.min_lng + ((x - self.origin_x) / self.svg_w) * self.geo_w
        lat = self.max_lat - ((y - self.origin_y) / self.svg_h) * self.geo_h
        return (round(lng, 8), round(lat, 8))


# ── SVG Parser ───────────────────────────────────────────────────────────────

class SvgParser:
    """Parses an inMapper SVG and extracts all geometric features."""

    ROOM_GROUPS = [
        "Walking", "Building", "Stand", "Service", "Food",
        "Water", "Other", "Shop", "Green", "Medical",
        "Commercial", "Social",
    ]

    def __init__(self, svg_path: str):
        self.tree = ET.parse(svg_path)
        self.root = self.tree.getroot()

        vb = self.root.get("viewBox", "0 0 100 100").split()
        self.vb_x, self.vb_y = float(vb[0]), float(vb[1])
        self.width = float(vb[2])
        self.height = float(vb[3])

    def _find_group(self, gid: str):
        g = self.root.find(f".//svg:g[@id='{gid}']", NS)
        if g is None:
            for elem in self.root.iter(f"{{{SVG_NS}}}g"):
                if elem.get(f"{{{INKSCAPE_NS}}}label") == gid:
                    return elem
        return g

    @staticmethod
    def _parse_transform(elem) -> Tuple[float, float]:
        """Extract translate(tx, ty) from a group's transform attribute."""
        t = elem.get("transform", "") if elem is not None else ""
        m = re.search(r"translate\(\s*([-\d.eE+]+)[,\s]+([-\d.eE+]+)\s*\)", t)
        if m:
            return float(m.group(1)), float(m.group(2))
        return 0.0, 0.0

    def _apply_tx(
        self, coords: List[Tuple[float, float]], tx: float, ty: float
    ) -> List[Tuple[float, float]]:
        if tx == 0.0 and ty == 0.0:
            return coords
        return [(x + tx, y + ty) for x, y in coords]

    # ── Room polygons ──

    def parse_rooms(self) -> Dict[str, List[dict]]:
        rooms_g = self._find_group("Rooms")
        if rooms_g is None:
            return {}

        self._rooms_tx, self._rooms_ty = self._parse_transform(rooms_g)
        tx, ty = self._rooms_tx, self._rooms_ty

        result: Dict[str, List[dict]] = {}

        # Subgroups (Building, Walking, Stand, Shop, etc.)
        for child_g in rooms_g.findall("./svg:g", NS):
            layer_name = child_g.get("id") or child_g.get(f"{{{INKSCAPE_NS}}}label")
            if not layer_name:
                continue

            fill = child_g.get("fill")
            stroke = child_g.get("stroke")
            items = []

            for path_el in child_g.findall("./svg:path", NS):
                d = path_el.get("d")
                pid = path_el.get("id", "unknown")
                if not d:
                    continue
                coords = path_to_absolute_coords(d)
                if len(coords) < 3:
                    continue
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                etx, ety = self._parse_transform(path_el)
                coords = self._apply_tx(coords, etx + tx, ety + ty)

                sw = path_el.get("stroke-width")
                p_fill = path_el.get("fill")
                items.append({
                    "id": pid,
                    "coords": coords,
                    "fill": p_fill or fill,
                    "stroke": stroke,
                    "stroke_width": sw,
                })

            if items:
                result[layer_name] = items

        # Direct children of Rooms (not inside any subgroup)
        structure_items = []
        for child in rooms_g:
            tag = child.tag.split("}")[-1]
            if tag == "g":
                continue
            cid = child.get("id", "unknown")

            if tag == "path":
                d = child.get("d")
                if not d:
                    continue
                coords = path_to_absolute_coords(d)
                if len(coords) >= 3:
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    etx, ety = self._parse_transform(child)
                    coords = self._apply_tx(coords, etx + tx, ety + ty)
                    structure_items.append({
                        "id": cid,
                        "coords": coords,
                        "fill": child.get("fill"),
                        "stroke": child.get("stroke", "#969696"),
                        "stroke_width": child.get("stroke-width"),
                    })
            elif tag == "circle":
                cx_val = float(child.get("cx", 0))
                cy_val = float(child.get("cy", 0))
                r = float(child.get("r", 0))
                n = 24
                coords = [
                    (cx_val + r * math.cos(2 * math.pi * i / n),
                     cy_val + r * math.sin(2 * math.pi * i / n))
                    for i in range(n)
                ]
                coords.append(coords[0])
                etx, ety = self._parse_transform(child)
                coords = self._apply_tx(coords, etx + tx, ety + ty)
                structure_items.append({
                    "id": cid,
                    "coords": coords,
                    "fill": child.get("fill", "#f5f5f5"),
                    "stroke": child.get("stroke", "#969696"),
                    "stroke_width": child.get("stroke-width"),
                })

        if structure_items:
            result.setdefault("Structure", []).extend(structure_items)

        return result

    # ── Paths / Doors / Portals (line segments) ──

    def _parse_line_group(self, group_id: str) -> List[dict]:
        g = self._find_group(group_id)
        if g is None:
            return []

        tx, ty = self._parse_transform(g)

        segments = []
        for el in g:
            tag = el.tag.split("}")[-1]
            eid = el.get("id", "")
            stroke = el.get("stroke")

            etx, ety = self._parse_transform(el)
            total_tx = etx + tx
            total_ty = ety + ty

            if tag == "line":
                x1 = float(el.get("x1", 0)) + total_tx
                y1 = float(el.get("y1", 0)) + total_ty
                x2 = float(el.get("x2", 0)) + total_tx
                y2 = float(el.get("y2", 0)) + total_ty
                segments.append({"id": eid, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "stroke": stroke})
            elif tag == "path":
                d = el.get("d")
                if not d:
                    continue
                lc = path_to_line_coords(d)
                if lc:
                    segments.append({"id": eid, "x1": lc[0] + total_tx, "y1": lc[1] + total_ty, "x2": lc[2] + total_tx, "y2": lc[3] + total_ty, "stroke": stroke})

        return segments

    def parse_paths(self) -> List[dict]:
        return self._parse_line_group("Paths")

    def parse_doors(self) -> List[dict]:
        return self._parse_line_group("Doors")

    def parse_portals(self) -> List[dict]:
        return self._parse_line_group("Portals")

    # ── Writing (text labels) ──

    def parse_writing(self) -> List[dict]:
        g = self._find_group("Writing")
        if g is None:
            return []

        tx, ty = self._parse_transform(g)
        transform = g.get("transform", "")
        rotation = 0.0
        m = re.search(r"rotate\(([-\d.]+)", transform)
        if m:
            rotation = float(m.group(1))

        labels = []
        for txt in g.findall("./svg:text", NS):
            x = float(txt.get("x", 0))
            y = float(txt.get("y", 0))
            fs = txt.get("font-size", "12")
            tspans = txt.findall("./svg:tspan", NS)
            lines = [ts.text or "" for ts in tspans]
            tid = txt.get("id", "")

            if rotation != 0.0:
                rad = math.radians(rotation)
                rx = x * math.cos(rad) - y * math.sin(rad)
                ry = x * math.sin(rad) + y * math.cos(rad)
                x, y = rx, ry

            etx, ety = self._parse_transform(txt)
            x += etx + tx
            y += ety + ty

            labels.append({
                "id": tid,
                "x": x, "y": y,
                "text": "\n".join(lines),
                "lines": lines,
                "font_size": fs,
                "rotation": rotation,
            })

        return labels

    # ── Icons & Constants ──

    def _parse_icon_group(self, group_id: str) -> List[dict]:
        g = self._find_group(group_id)
        if g is None:
            return []

        tx, ty = self._parse_transform(g)

        icons = []
        for child in g:
            tag = child.tag.split("}")[-1]
            cid = child.get("id", "")

            if tag == "g":
                paths = child.findall(f".//{{{SVG_NS}}}path")
                all_coords = []
                for p in paths:
                    d = p.get("d")
                    if d:
                        try:
                            all_coords.extend(path_to_absolute_coords(d))
                        except Exception:
                            pass
                if all_coords:
                    cx = sum(c[0] for c in all_coords) / len(all_coords) + tx
                    cy = sum(c[1] for c in all_coords) / len(all_coords) + ty
                    icons.append({"id": cid, "x": cx, "y": cy, "type": group_id.lower()})
            elif tag == "path":
                d = child.get("d", "")
                if d:
                    try:
                        coords = path_to_absolute_coords(d)
                        if coords:
                            cx = sum(c[0] for c in coords) / len(coords) + tx
                            cy = sum(c[1] for c in coords) / len(coords) + ty
                            icons.append({"id": cid, "x": cx, "y": cy, "type": group_id.lower()})
                    except Exception:
                        pass

        return icons

    def parse_icons(self) -> List[dict]:
        return self._parse_icon_group("Icons")

    def parse_constants(self) -> List[dict]:
        return self._parse_icon_group("Constants")


# ── GeoJSON Builder ──────────────────────────────────────────────────────────

def _bbox_overlap_ratio(coords, extent_min_x, extent_min_y, extent_max_x, extent_max_y):
    """Return fraction of polygon bbox that overlaps with the given extent."""
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
    bw = bx1 - bx0
    bh = by1 - by0
    if bw <= 0 or bh <= 0:
        return 0.0
    ox0 = max(bx0, extent_min_x)
    oy0 = max(by0, extent_min_y)
    ox1 = min(bx1, extent_max_x)
    oy1 = min(by1, extent_max_y)
    if ox1 <= ox0 or oy1 <= oy0:
        return 0.0
    return ((ox1 - ox0) * (oy1 - oy0)) / (bw * bh)


def build_geojson(parser: SvgParser, transform: GeoTransform) -> dict:
    features = []

    content_max_x = transform.origin_x + transform.svg_w
    content_max_y = transform.origin_y + transform.svg_h
    margin_x = transform.svg_w * 0.1
    margin_y = transform.svg_h * 0.1

    # Room polygons
    rooms = parser.parse_rooms()
    for layer_name, items in rooms.items():
        is_walking = layer_name.lower() == "walking"
        for item in items:
            if is_walking:
                ratio = _bbox_overlap_ratio(
                    item["coords"],
                    transform.origin_x - margin_x,
                    transform.origin_y - margin_y,
                    content_max_x + margin_x,
                    content_max_y + margin_y,
                )
                if ratio < 0.05:
                    continue

            ring = [transform.to_lnglat(x, y) for x, y in item["coords"]]
            features.append({
                "type": "Feature",
                "properties": {
                    "id": item["id"],
                    "layer": "rooms",
                    "sublayer": layer_name.lower(),
                    "fill": item.get("fill"),
                    "stroke": item.get("stroke"),
                    "stroke_width": item.get("stroke_width"),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring],
                },
            })

    # Paths
    for seg in parser.parse_paths():
        p1 = transform.to_lnglat(seg["x1"], seg["y1"])
        p2 = transform.to_lnglat(seg["x2"], seg["y2"])
        features.append({
            "type": "Feature",
            "properties": {
                "id": seg["id"],
                "layer": "paths",
                "stroke": seg.get("stroke"),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [p1, p2],
            },
        })

    # Doors
    for seg in parser.parse_doors():
        p1 = transform.to_lnglat(seg["x1"], seg["y1"])
        p2 = transform.to_lnglat(seg["x2"], seg["y2"])
        features.append({
            "type": "Feature",
            "properties": {
                "id": seg["id"],
                "layer": "doors",
                "stroke": seg.get("stroke"),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [p1, p2],
            },
        })

    # Portals
    for seg in parser.parse_portals():
        p1 = transform.to_lnglat(seg["x1"], seg["y1"])
        p2 = transform.to_lnglat(seg["x2"], seg["y2"])
        features.append({
            "type": "Feature",
            "properties": {
                "id": seg["id"],
                "layer": "portals",
                "stroke": seg.get("stroke"),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [p1, p2],
            },
        })

    # Writing labels
    for lbl in parser.parse_writing():
        pt = transform.to_lnglat(lbl["x"], lbl["y"])
        features.append({
            "type": "Feature",
            "properties": {
                "id": lbl["id"],
                "layer": "writing",
                "text": lbl["text"],
                "lines": lbl["lines"],
                "font_size": lbl["font_size"],
            },
            "geometry": {
                "type": "Point",
                "coordinates": pt,
            },
        })

    # Icons
    for icon in parser.parse_icons():
        pt = transform.to_lnglat(icon["x"], icon["y"])
        features.append({
            "type": "Feature",
            "properties": {
                "id": icon["id"],
                "layer": "icons",
                "type": icon.get("type"),
            },
            "geometry": {
                "type": "Point",
                "coordinates": pt,
            },
        })

    # Constants
    for const in parser.parse_constants():
        pt = transform.to_lnglat(const["x"], const["y"])
        features.append({
            "type": "Feature",
            "properties": {
                "id": const["id"],
                "layer": "constants",
                "type": const.get("type"),
            },
            "geometry": {
                "type": "Point",
                "coordinates": pt,
            },
        })

    return {"type": "FeatureCollection", "features": features}


# ── MapLibre GL JS Viewer ────────────────────────────────────────────────────

def generate_viewer_html(geojson_path: str, output_html: str, center_lng: float, center_lat: float):
    geojson_filename = os.path.basename(geojson_path)

    sublayer_colors = {
        "walking": "#f5f5f5",
        "building": "#e6e6e6",
        "stand": "#d9d3d2",
        "service": "#e9dad0",
        "food": "#d1bbbc",
        "water": "#cfe2f3",
        "other": "#e9dad0",
        "shop": "#d9d3d2",
        "green": "#a8d08d",
        "medical": "#ff9999",
        "commercial": "#ffe0b2",
        "social": "#c5cae9",
        "structure": "#d0d0d0",
    }

    sublayer_heights = {
        "walking": 0,
        "building": 0,
        "stand": 8,
        "service": 6,
        "food": 6,
        "water": 0.5,
        "other": 5,
        "shop": 8,
        "green": 1,
        "medical": 6,
        "commercial": 7,
        "social": 5,
        "structure": 3,
    }

    color_match_expr = ["match", ["get", "sublayer"]]
    for sl, color in sublayer_colors.items():
        color_match_expr.extend([sl, color])
    color_match_expr.append("#cccccc")

    height_match_expr = ["match", ["get", "sublayer"]]
    for sl, h in sublayer_heights.items():
        height_match_expr.extend([sl, h])
    height_match_expr.append(4)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>inMapper GeoJSON Viewer</title>
<script src="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js"></script>
<link href="https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css" rel="stylesheet">
<style>
  body {{ margin: 0; padding: 0; }}
  #map {{ position: absolute; top: 0; bottom: 0; width: 100%; }}
  #info {{
    position: absolute; top: 10px; left: 10px; z-index: 1;
    background: rgba(255,255,255,0.92); padding: 12px 16px;
    border-radius: 8px; font-family: 'Segoe UI', sans-serif; font-size: 13px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.18);
    max-width: 220px;
  }}
  #info h3 {{ margin: 0 0 8px; font-size: 15px; color: #333; }}
  .layer-toggle {{ display: flex; align-items: center; cursor: pointer; margin: 3px 0; }}
  .layer-toggle input {{ margin-right: 8px; accent-color: #4a90d9; }}
  .separator {{ border-top: 1px solid #e0e0e0; margin: 8px 0; }}
  .slider-group {{ margin: 4px 0; }}
  .slider-group label {{ font-size: 12px; color: #666; display: block; margin-bottom: 2px; }}
  .slider-group input[type=range] {{ width: 100%; }}
  #hover-info {{ margin-top: 8px; color: #555; min-height: 18px; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="info">
  <h3>inMapper GeoJSON</h3>
  <div id="layers"></div>
  <div class="separator"></div>
  <div class="slider-group">
    <label>Pitch: <span id="pitch-val">60</span>&deg;</label>
    <input type="range" id="pitch-slider" min="0" max="85" value="60">
  </div>
  <div class="slider-group">
    <label>Height scale: <span id="height-val">1.0</span>x</label>
    <input type="range" id="height-slider" min="0" max="30" value="10" step="1">
  </div>
  <div id="hover-info"></div>
</div>
<script>
const GEOJSON_URL = '{geojson_filename}';

const SUBLAYER_COLORS = {json.dumps(sublayer_colors)};
const SUBLAYER_HEIGHTS = {json.dumps(sublayer_heights)};

let heightScale = 1.0;
let roomData = null;

const map = new maplibregl.Map({{
  container: 'map',
  style: {{
    version: 8,
    sources: {{}},
    layers: [{{
      id: 'background',
      type: 'background',
      paint: {{ 'background-color': '#f0f0f0' }}
    }}],
    light: {{
      anchor: 'viewport',
      color: '#ffffff',
      intensity: 0.4,
      position: [1.5, 180, 30]
    }}
  }},
  center: [{center_lng}, {center_lat}],
  zoom: 16,
  pitch: 60,
  bearing: -20,
  maxZoom: 24,
  antialias: true,
}});

map.addControl(new maplibregl.NavigationControl({{ visualizePitch: true }}));

document.getElementById('pitch-slider').addEventListener('input', function() {{
  map.setPitch(Number(this.value));
  document.getElementById('pitch-val').textContent = this.value;
}});

document.getElementById('height-slider').addEventListener('input', function() {{
  heightScale = Number(this.value) / 10;
  document.getElementById('height-val').textContent = heightScale.toFixed(1);
  updateExtrusionHeights();
}});

function updateExtrusionHeights() {{
  if (!map.getLayer('rooms-3d')) return;
  const expr = ['match', ['get', 'sublayer']];
  for (const [sl, h] of Object.entries(SUBLAYER_HEIGHTS)) {{
    expr.push(sl, h * heightScale);
  }}
  expr.push(4 * heightScale);
  map.setPaintProperty('rooms-3d', 'fill-extrusion-height', expr);
}}

map.on('load', async () => {{
  const resp = await fetch(GEOJSON_URL);
  const data = await resp.json();

  const layerNames = ['rooms', 'paths', 'doors', 'portals', 'writing', 'icons', 'constants'];
  const layerData = {{}};
  layerNames.forEach(l => {{
    layerData[l] = {{
      type: 'FeatureCollection',
      features: data.features.filter(f => f.properties.layer === l)
    }};
  }});
  roomData = layerData['rooms'];

  layerNames.forEach(l => {{
    map.addSource(l, {{ type: 'geojson', data: layerData[l] }});
  }});

  // Flat base layers: Walking + Building (no extrusion)
  const flatSublayers = ['walking', 'building'];
  map.addSource('rooms-flat', {{
    type: 'geojson',
    data: {{
      type: 'FeatureCollection',
      features: roomData.features.filter(f => flatSublayers.includes(f.properties.sublayer))
    }}
  }});
  map.addLayer({{
    id: 'rooms-floor',
    type: 'fill',
    source: 'rooms-flat',
    paint: {{
      'fill-color': {json.dumps(color_match_expr)},
      'fill-opacity': 0.9,
    }}
  }});

  // 3D room extrusions (only units: shops, food, service, etc.)
  map.addLayer({{
    id: 'rooms-3d',
    type: 'fill-extrusion',
    source: 'rooms',
    filter: ['all',
      ['!=', ['get', 'sublayer'], 'walking'],
      ['!=', ['get', 'sublayer'], 'building']
    ],
    paint: {{
      'fill-extrusion-color': {json.dumps(color_match_expr)},
      'fill-extrusion-height': {json.dumps(height_match_expr)},
      'fill-extrusion-base': 0,
      'fill-extrusion-opacity': 0.88,
    }}
  }});

  // Room outlines
  map.addLayer({{
    id: 'rooms-outline',
    type: 'line',
    source: 'rooms',
    paint: {{
      'line-color': '#888888',
      'line-width': 1,
    }}
  }});

  // Paths
  map.addLayer({{
    id: 'paths-line',
    type: 'line',
    source: 'paths',
    paint: {{
      'line-color': '#3fab35',
      'line-width': 1.5,
    }},
    layout: {{ visibility: 'none' }}
  }});

  // Doors
  map.addLayer({{
    id: 'doors-line',
    type: 'line',
    source: 'doors',
    paint: {{
      'line-color': '#ff0000',
      'line-width': 1.5,
    }},
    layout: {{ visibility: 'none' }}
  }});

  // Portals
  map.addLayer({{
    id: 'portals-line',
    type: 'line',
    source: 'portals',
    paint: {{
      'line-color': '#0000ff',
      'line-width': 2,
    }},
    layout: {{ visibility: 'none' }}
  }});

  // Writing labels
  map.addLayer({{
    id: 'writing-labels',
    type: 'symbol',
    source: 'writing',
    layout: {{
      'text-field': ['get', 'text'],
      'text-size': 10,
      'text-anchor': 'center',
      'text-allow-overlap': true,
    }},
    paint: {{
      'text-color': '#3f3a38',
      'text-halo-color': '#ffffff',
      'text-halo-width': 1,
    }}
  }});

  // Icons
  map.addLayer({{
    id: 'icons-circle',
    type: 'circle',
    source: 'icons',
    paint: {{
      'circle-radius': 5,
      'circle-color': '#2196F3',
      'circle-stroke-width': 1.5,
      'circle-stroke-color': '#fff',
    }}
  }});

  // Constants
  map.addLayer({{
    id: 'constants-circle',
    type: 'circle',
    source: 'constants',
    paint: {{
      'circle-radius': 6,
      'circle-color': '#FF9800',
      'circle-stroke-width': 1.5,
      'circle-stroke-color': '#fff',
    }}
  }});

  // Layer toggles
  const toggles = [
    {{ name: 'Rooms (3D)', layers: ['rooms-3d', 'rooms-outline', 'rooms-floor'], checked: true }},
    {{ name: 'Paths', layers: ['paths-line'], checked: false }},
    {{ name: 'Doors', layers: ['doors-line'], checked: false }},
    {{ name: 'Portals', layers: ['portals-line'], checked: false }},
    {{ name: 'Writing', layers: ['writing-labels'], checked: true }},
    {{ name: 'Icons', layers: ['icons-circle'], checked: true }},
    {{ name: 'Constants', layers: ['constants-circle'], checked: true }},
  ];

  const layersDiv = document.getElementById('layers');
  toggles.forEach(t => {{
    const label = document.createElement('label');
    label.className = 'layer-toggle';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = t.checked;
    cb.onchange = () => {{
      const vis = cb.checked ? 'visible' : 'none';
      t.layers.forEach(lid => {{
        if (map.getLayer(lid)) map.setLayoutProperty(lid, 'visibility', vis);
      }});
    }};
    label.appendChild(cb);
    label.appendChild(document.createTextNode(t.name));
    label.appendChild(document.createElement('br'));
    layersDiv.appendChild(label);
  }});

  // Hover info
  const hoverInfo = document.getElementById('hover-info');
  map.on('mousemove', 'rooms-3d', e => {{
    if (e.features.length) {{
      const p = e.features[0].properties;
      hoverInfo.innerHTML = '<b>' + (p.id || '') + '</b><br><span style="color:#888">' + (p.sublayer || '') + '</span>';
      map.getCanvas().style.cursor = 'pointer';
    }}
  }});
  map.on('mouseleave', 'rooms-3d', () => {{
    hoverInfo.innerHTML = '';
    map.getCanvas().style.cursor = '';
  }});

  // Fit to bounds
  const bounds = new maplibregl.LngLatBounds();
  data.features.forEach(f => {{
    const geom = f.geometry;
    if (geom.type === 'Point') bounds.extend(geom.coordinates);
    else if (geom.type === 'LineString') geom.coordinates.forEach(c => bounds.extend(c));
    else if (geom.type === 'Polygon') geom.coordinates[0].forEach(c => bounds.extend(c));
  }});
  map.fitBounds(bounds, {{ padding: 40, pitch: 60, bearing: -20 }});
}});
</script>
</body>
</html>"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Viewer: {output_html}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="inMapper SVG → GeoJSON converter")
    ap.add_argument("svg", help="Input SVG file")
    ap.add_argument("-o", "--output", help="Output GeoJSON file (default: <input>.geojson)")
    ap.add_argument("--center-lat", type=float, default=0.0, help="Center latitude (default: 0)")
    ap.add_argument("--center-lng", type=float, default=0.0, help="Center longitude (default: 0)")
    ap.add_argument("--scale", type=float, default=0.03,
                    help="Meters per SVG unit (default: 0.03)")
    ap.add_argument("--viewer", action="store_true", help="Generate MapLibre GL JS HTML viewer")
    args = ap.parse_args()

    if not os.path.isfile(args.svg):
        print(f"Error: {args.svg} not found")
        sys.exit(1)

    out_path = args.output or os.path.splitext(args.svg)[0] + ".geojson"

    print(f"Parsing SVG: {args.svg}")
    parser = SvgParser(args.svg)
    print(f"  viewBox: 0 0 {parser.width} {parser.height}")

    # Compute content extent from non-Walking rooms + paths
    all_xs, all_ys = [], []
    rooms = parser.parse_rooms()
    for layer_name, items in rooms.items():
        if layer_name.lower() == "walking":
            continue
        for item in items:
            for x, y in item["coords"]:
                all_xs.append(x)
                all_ys.append(y)
    for seg in parser.parse_paths():
        all_xs.extend([seg["x1"], seg["x2"]])
        all_ys.extend([seg["y1"], seg["y2"]])

    if all_xs and all_ys:
        content_ox = min(all_xs)
        content_oy = min(all_ys)
        content_w = max(all_xs) - content_ox
        content_h = max(all_ys) - content_oy
        print(f"  Content extent (excl. Walking): ({content_ox:.1f},{content_oy:.1f}) - ({content_ox+content_w:.1f},{content_oy+content_h:.1f})")
    else:
        content_w = parser.width
        content_h = parser.height
        content_ox = 0
        content_oy = 0

    transform = GeoTransform(
        content_w, content_h,
        center_lat=args.center_lat,
        center_lng=args.center_lng,
        scale=args.scale,
        origin_x=content_ox,
        origin_y=content_oy,
    )
    print(f"  Geo extent: {transform.geo_w:.6f}° lng × {transform.geo_h:.6f}° lat")

    geojson = build_geojson(parser, transform)

    layer_counts = {}
    for f in geojson["features"]:
        l = f["properties"]["layer"]
        layer_counts[l] = layer_counts.get(l, 0) + 1
    print(f"  Features: {len(geojson['features'])} total")
    for l, c in sorted(layer_counts.items()):
        print(f"    {l}: {c}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)
    print(f"  Output: {out_path}")

    if args.viewer:
        html_path = os.path.splitext(out_path)[0] + "_viewer.html"
        generate_viewer_html(out_path, html_path, args.center_lng, args.center_lat)

    print("Done!")


if __name__ == "__main__":
    main()
