"""
Automatic Path & Door Generator for inMapper SVGs (v2)
======================================================
Generates corridor centerlines using Voronoi diagram of booth edges,
producing clean, grid-aligned path networks.

Usage:
    py tools/auto_pathgen.py tools/fuar.svg -o tools/fuar_routed.svg
    py tools/auto_pathgen.py input.svg -o output.svg --sample-spacing 3 --door-len 8
"""

import argparse
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import (
    Polygon, MultiPolygon, Point, LineString, box,
)
from shapely.ops import unary_union
from shapely.prepared import prep

# ── SVG Namespace helpers ────────────────────────────────────────────────

SVG_NS = "http://www.w3.org/2000/svg"
INKSCAPE_NS = "http://www.inkscape.org/namespaces/inkscape"
NS = {"svg": SVG_NS, "inkscape": INKSCAPE_NS}

BOOTH_SUBLAYERS = {
    "Stand", "Service", "Food", "Shop", "Other",
    "Water", "Green", "Medical", "Commercial", "Social",
}


# ── SVG Path Parsing (robust, handles all commands) ──────────────────────

def _tokenize_path(d: str):
    tokens = re.findall(
        r"[a-df-zA-DF-Z]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", d
    )
    commands = []
    i = 0
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]; i += 1
            coords = []
            while i < len(tokens) and not tokens[i].isalpha():
                coords.append(float(tokens[i])); i += 1
            commands.append((cmd, coords))
        else:
            i += 1
    return commands


def path_to_absolute_coords(d: str) -> list[tuple[float, float]]:
    commands = _tokenize_path(d)
    pts: list[tuple[float, float]] = []
    cx, cy = 0.0, 0.0
    sx, sy = 0.0, 0.0
    for cmd, coords in commands:
        if cmd == "m":
            if not pts:
                cx, cy = coords[0], coords[1]
            else:
                cx += coords[0]; cy += coords[1]
            sx, sy = cx, cy; pts.append((cx, cy))
            for i in range(2, len(coords), 2):
                cx += coords[i]; cy += coords[i + 1]; pts.append((cx, cy))
        elif cmd == "M":
            cx, cy = coords[0], coords[1]
            sx, sy = cx, cy; pts.append((cx, cy))
            for i in range(2, len(coords), 2):
                cx, cy = coords[i], coords[i + 1]; pts.append((cx, cy))
        elif cmd == "l":
            for i in range(0, len(coords), 2):
                cx += coords[i]; cy += coords[i + 1]; pts.append((cx, cy))
        elif cmd == "L":
            for i in range(0, len(coords), 2):
                cx, cy = coords[i], coords[i + 1]; pts.append((cx, cy))
        elif cmd == "h":
            for v in coords: cx += v; pts.append((cx, cy))
        elif cmd == "H":
            for v in coords: cx = v; pts.append((cx, cy))
        elif cmd == "v":
            for v in coords: cy += v; pts.append((cx, cy))
        elif cmd == "V":
            for v in coords: cy = v; pts.append((cx, cy))
        elif cmd in ("z", "Z"):
            if pts and (cx != sx or cy != sy):
                pts.append((sx, sy))
            cx, cy = sx, sy
    return pts


def _parse_transform(elem) -> tuple[float, float]:
    t = elem.get("transform", "") if elem is not None else ""
    m = re.search(r"translate\(\s*([-\d.eE+]+)[,\s]+([-\d.eE+]+)\s*\)", t)
    if m:
        return float(m.group(1)), float(m.group(2))
    return 0.0, 0.0


# ── SVG Parsing ──────────────────────────────────────────────────────────

@dataclass
class BoothPoly:
    svg_id: str
    polygon: Polygon
    sublayer: str


def parse_svg(svg_path: str):
    """Parse SVG and extract booth polygons, walking area, and building boundary."""
    tree = ET.parse(svg_path)
    root = tree.getroot()

    vb = root.get("viewBox", "0 0 1000 1000").split()
    svg_w, svg_h = float(vb[2]), float(vb[3])

    rooms_g = root.find(".//svg:g[@id='Rooms']", NS)
    if rooms_g is None:
        for elem in root.iter(f"{{{SVG_NS}}}g"):
            if elem.get(f"{{{INKSCAPE_NS}}}label") == "Rooms":
                rooms_g = elem
                break

    if rooms_g is None:
        print("ERROR: Rooms group not found!")
        return tree, [], None, None, svg_w, svg_h

    rooms_tx, rooms_ty = _parse_transform(rooms_g)

    booths: list[BoothPoly] = []
    walking_polys: list[Polygon] = []
    building_polys: list[Polygon] = []

    for child_g in rooms_g.findall("./svg:g", NS):
        layer_name = child_g.get("id") or child_g.get(f"{{{INKSCAPE_NS}}}label", "")
        if not layer_name:
            continue

        for path_el in child_g.findall("./svg:path", NS):
            d = path_el.get("d", "")
            pid = path_el.get("id", "")
            if not d:
                continue

            coords = path_to_absolute_coords(d)
            if len(coords) < 3:
                continue
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            etx, ety = _parse_transform(path_el)
            ttx, tty = etx + rooms_tx, ety + rooms_ty
            if ttx != 0 or tty != 0:
                coords = [(x + ttx, y + tty) for x, y in coords]

            try:
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty or poly.area < 1:
                    continue
                if isinstance(poly, MultiPolygon):
                    polys = list(poly.geoms)
                else:
                    polys = [poly]
            except Exception:
                continue

            for p in polys:
                if layer_name == "Walking":
                    walking_polys.append(p)
                elif layer_name == "Building":
                    building_polys.append(p)
                elif layer_name in BOOTH_SUBLAYERS:
                    booths.append(BoothPoly(svg_id=pid, polygon=p, sublayer=layer_name))

    walking = unary_union(walking_polys) if walking_polys else None
    building = unary_union(building_polys) if building_polys else None

    return tree, booths, walking, building, svg_w, svg_h


# ── Voronoi Corridor Extraction ──────────────────────────────────────────

def _sample_polygon_edges(geom, spacing: float) -> list[tuple[float, float]]:
    """Sample evenly-spaced points along a polygon/multipolygon exterior."""
    if isinstance(geom, MultiPolygon):
        pts = []
        for poly in geom.geoms:
            pts.extend(_sample_polygon_edges(poly, spacing))
        return pts
    if not isinstance(geom, Polygon):
        return []
    ext = geom.exterior
    length = ext.length
    n = max(int(length / spacing), 4)
    return [(pt.x, pt.y) for pt in (ext.interpolate(i * spacing) for i in range(n))]


def compute_voronoi_corridors(
    booths: list[BoothPoly],
    boundary: Polygon | MultiPolygon,
    booth_spacing: float = 3.0,
    boundary_spacing: float = 10.0,
    min_corridor_width: float = 5.0,
) -> list[tuple[float, float, float, float]]:
    """Compute corridor centerlines via Voronoi diagram of booth edge samples."""

    sample_points: list[tuple[float, float]] = []

    for booth in booths:
        sample_points.extend(_sample_polygon_edges(booth.polygon, booth_spacing))

    if isinstance(boundary, MultiPolygon):
        for poly in boundary.geoms:
            sample_points.extend(_sample_polygon_edges(poly, boundary_spacing))
    elif isinstance(boundary, Polygon):
        sample_points.extend(_sample_polygon_edges(boundary, boundary_spacing))

    if len(sample_points) < 4:
        return []

    print(f"  Voronoi input: {len(sample_points)} sample points")

    pts_array = np.array(sample_points)
    vor = Voronoi(pts_array)

    booth_union = unary_union([b.polygon for b in booths])
    booth_buffered = unary_union([b.polygon.buffer(min_corridor_width * 0.3) for b in booths])
    prep_boundary = prep(boundary)
    prep_booths_buf = prep(booth_buffered)

    valid_segments = []

    for v1_idx, v2_idx in vor.ridge_vertices:
        if v1_idx < 0 or v2_idx < 0:
            continue

        v1 = vor.vertices[v1_idx]
        v2 = vor.vertices[v2_idx]

        p1, p2 = Point(v1[0], v1[1]), Point(v2[0], v2[1])

        if not prep_boundary.contains(p1) or not prep_boundary.contains(p2):
            continue

        if prep_booths_buf.contains(p1) or prep_booths_buf.contains(p2):
            continue

        edge_line = LineString([v1, v2])
        if edge_line.intersects(booth_union):
            continue

        seg_len = math.hypot(v2[0] - v1[0], v2[1] - v1[1])
        if seg_len < 0.5:
            continue

        valid_segments.append((
            round(v1[0], 2), round(v1[1], 2),
            round(v2[0], 2), round(v2[1], 2),
        ))

    print(f"  Valid Voronoi edges: {len(valid_segments)}")
    return valid_segments


# ── Graph Building & Cleanup ─────────────────────────────────────────────

def _build_graph(segments, tolerance=1.5):
    """Build adjacency graph, snapping nearby endpoints into single nodes."""
    nodes: list[tuple[float, float]] = []

    def find_or_add(x, y):
        for i, (nx, ny) in enumerate(nodes):
            if abs(nx - x) <= tolerance and abs(ny - y) <= tolerance:
                return i
        nodes.append((x, y))
        return len(nodes) - 1

    adj: dict[int, set[int]] = {}

    for x1, y1, x2, y2 in segments:
        n1 = find_or_add(x1, y1)
        n2 = find_or_add(x2, y2)
        if n1 == n2:
            continue
        adj.setdefault(n1, set()).add(n2)
        adj.setdefault(n2, set()).add(n1)

    return nodes, adj


def _prune_short_deadends(nodes, adj, min_length=10.0, iterations=8):
    """Iteratively remove degree-1 nodes connected by short edges."""
    for _ in range(iterations):
        to_remove = []
        for n in list(adj.keys()):
            nbs = adj.get(n, set())
            if len(nbs) != 1:
                continue
            nb = next(iter(nbs))
            d = math.hypot(nodes[n][0] - nodes[nb][0], nodes[n][1] - nodes[nb][1])
            if d < min_length:
                to_remove.append(n)
        if not to_remove:
            break
        for n in to_remove:
            for nb in list(adj.get(n, set())):
                adj[nb].discard(n)
            adj.pop(n, None)


def _merge_and_simplify(nodes, adj, simplify_tol=2.5):
    """Merge degree-2 chains into polylines, then simplify with Douglas-Peucker."""
    degree2 = {n for n in adj if len(adj.get(n, set())) == 2}
    visited_edges: set[tuple[int, int]] = set()
    chains: list[list[int]] = []

    for start in adj:
        if start in degree2:
            continue
        for nb in list(adj.get(start, set())):
            ek = (min(start, nb), max(start, nb))
            if ek in visited_edges:
                continue
            visited_edges.add(ek)

            chain = [start]
            current, prev = nb, start

            while current in degree2:
                chain.append(current)
                nbs = list(adj[current])
                nxt = nbs[0] if nbs[1] == prev else nbs[1]
                ek2 = (min(current, nxt), max(current, nxt))
                visited_edges.add(ek2)
                prev, current = current, nxt

            chain.append(current)
            chains.append(chain)

    segments: list[tuple[float, float, float, float]] = []
    for chain in chains:
        if len(chain) < 2:
            continue
        coords = [nodes[n] for n in chain]
        line = LineString(coords)
        simplified = line.simplify(simplify_tol)
        sc = list(simplified.coords)
        for i in range(len(sc) - 1):
            segments.append((round(sc[i][0], 2), round(sc[i][1], 2),
                             round(sc[i + 1][0], 2), round(sc[i + 1][1], 2)))
    return segments


def _straighten_segments(segments, angle_threshold_deg=10.0):
    """Snap nearly-H/V segments to exact horizontal or vertical,
    then reconnect broken joints."""
    result = []
    for x1, y1, x2, y2 in segments:
        dx, dy = x2 - x1, y2 - y1
        angle = math.degrees(math.atan2(abs(dy), abs(dx)))
        if angle < angle_threshold_deg:
            mid_y = round((y1 + y2) / 2, 2)
            result.append((x1, mid_y, x2, mid_y))
        elif angle > (90 - angle_threshold_deg):
            mid_x = round((x1 + x2) / 2, 2)
            result.append((mid_x, y1, mid_x, y2))
        else:
            result.append((x1, y1, x2, y2))

    result = _reconnect_endpoints(result)
    return result


def _reconnect_endpoints(segments, tolerance=3.0):
    """After straightening, snap nearby endpoints so joints stay connected."""
    pts: list[list[float]] = []
    seg_refs: list[list[tuple[int, int]]] = []

    def find_or_add(x, y):
        for i, p in enumerate(pts):
            if abs(p[0] - x) <= tolerance and abs(p[1] - y) <= tolerance:
                return i
        pts.append([x, y])
        seg_refs.append([])
        return len(pts) - 1

    indexed = []
    for si, (x1, y1, x2, y2) in enumerate(segments):
        i1 = find_or_add(x1, y1)
        i2 = find_or_add(x2, y2)
        seg_refs[i1].append((si, 0))
        seg_refs[i2].append((si, 1))
        indexed.append([x1, y1, x2, y2])

    for pi, refs in enumerate(seg_refs):
        if len(refs) < 2:
            continue
        avg_x = sum(indexed[si][ei * 2] for si, ei in refs) / len(refs)
        avg_y = sum(indexed[si][ei * 2 + 1] for si, ei in refs) / len(refs)
        avg_x, avg_y = round(avg_x, 2), round(avg_y, 2)
        for si, ei in refs:
            indexed[si][ei * 2] = avg_x
            indexed[si][ei * 2 + 1] = avg_y

    return [(s[0], s[1], s[2], s[3]) for s in indexed]


def _validate_no_booth_intersection(segments, booths):
    """Remove segments that intersect any booth polygon."""
    booth_union = unary_union([b.polygon for b in booths])
    valid = []
    removed = 0
    for seg in segments:
        line = LineString([(seg[0], seg[1]), (seg[2], seg[3])])
        if not line.intersects(booth_union):
            valid.append(seg)
        else:
            removed += 1
    if removed:
        print(f"  Removed {removed} segments intersecting booths")
    return valid


def _remove_duplicate_segments(segments, tolerance=1.0):
    """Remove duplicate or near-duplicate segments."""
    unique = []
    for seg in segments:
        is_dup = False
        for u in unique:
            d1 = math.hypot(seg[0] - u[0], seg[1] - u[1]) + math.hypot(seg[2] - u[2], seg[3] - u[3])
            d2 = math.hypot(seg[0] - u[2], seg[1] - u[3]) + math.hypot(seg[2] - u[0], seg[3] - u[1])
            if min(d1, d2) < tolerance:
                is_dup = True
                break
        if not is_dup:
            unique.append(seg)
    return unique


def process_paths(raw_segments, booths, min_deadend=10.0, simplify_tol=2.5, angle_thr=10.0):
    """Full pipeline: graph -> prune -> merge -> straighten -> validate."""
    nodes, adj = _build_graph(raw_segments)
    print(f"  Graph: {len(nodes)} nodes, {sum(len(v) for v in adj.values()) // 2} edges")

    _prune_short_deadends(nodes, adj, min_length=min_deadend)
    live = {n for n in adj if adj.get(n)}
    print(f"  After pruning dead-ends: {len(live)} nodes")

    segments = _merge_and_simplify(nodes, adj, simplify_tol=simplify_tol)
    print(f"  After merge/simplify: {len(segments)} segments")

    segments = _straighten_segments(segments, angle_threshold_deg=angle_thr)
    segments = _validate_no_booth_intersection(segments, booths)
    segments = _remove_duplicate_segments(segments)
    print(f"  Final path segments: {len(segments)}")
    return segments


# ── Door Generation with T-junction splitting ───────────────────────────

def _project_point_on_segment(px, py, x1, y1, x2, y2):
    """Project point (px,py) onto segment (x1,y1)-(x2,y2).
    Returns (proj_x, proj_y, t) where t in [0,1] is the parameter along the segment."""
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-10:
        return x1, y1, 0.0
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    return x1 + t * dx, y1 + t * dy, t


def generate_doors(
    booths: list[BoothPoly],
    path_segments: list[tuple[float, float, float, float]],
    door_len: float = 8.0,
    min_corridor_width: float = 5.0,
):
    """Generate doors with T-junction: split the nearest path segment at the
    perpendicular projection point, then connect door to that split node.
    Returns (updated_path_segments, doors)."""
    if not path_segments:
        return path_segments, []

    working_segs = list(path_segments)
    doors: list[tuple[str, float, float, float, float]] = []

    for booth in booths:
        poly = booth.polygon
        if not poly.is_valid or poly.is_empty:
            continue

        exterior = list(poly.exterior.coords)
        if len(exterior) < 3:
            continue

        # Find the booth edge midpoint closest to any path segment
        best_dist = float("inf")
        best_edge_mid = None
        best_seg_idx = -1
        best_proj = None

        for i in range(len(exterior) - 1):
            ax, ay = exterior[i]
            bx, by = exterior[i + 1]
            elen = math.hypot(bx - ax, by - ay)
            if elen < 1.0:
                continue
            mx, my = (ax + bx) / 2, (ay + by) / 2

            for si, (sx1, sy1, sx2, sy2) in enumerate(working_segs):
                prx, pry, t = _project_point_on_segment(mx, my, sx1, sy1, sx2, sy2)
                d = math.hypot(mx - prx, my - pry)
                if 0.5 < d < best_dist:
                    best_dist = d
                    best_edge_mid = (mx, my)
                    best_seg_idx = si
                    best_proj = (round(prx, 2), round(pry, 2), t)

        if best_edge_mid is None or best_seg_idx < 0:
            continue

        if best_dist < min_corridor_width * 0.3:
            continue

        prx, pry, t = best_proj
        sx1, sy1, sx2, sy2 = working_segs[best_seg_idx]

        # Split the path segment at the projection point (T-junction)
        eps = 0.02
        if t > eps and t < (1.0 - eps):
            working_segs[best_seg_idx] = (sx1, sy1, prx, pry)
            working_segs.append((prx, pry, sx2, sy2))
        elif t <= eps:
            prx, pry = sx1, sy1
        else:
            prx, pry = sx2, sy2

        # Door from booth edge midpoint to the T-junction point on the path
        inner_x, inner_y = round(best_edge_mid[0], 2), round(best_edge_mid[1], 2)
        dx, dy = prx - inner_x, pry - inner_y
        dist = math.hypot(dx, dy)
        if dist < 0.5:
            continue

        if dist <= door_len * 1.5:
            outer_x, outer_y = prx, pry
        else:
            ratio = door_len / dist
            outer_x = round(inner_x + dx * ratio, 2)
            outer_y = round(inner_y + dy * ratio, 2)

        doors.append((f"{booth.svg_id}_1_", inner_x, inner_y, outer_x, outer_y))

    return working_segs, doors


# ── SVG Writer ───────────────────────────────────────────────────────────

def inject_paths_and_doors(
    tree: ET.ElementTree,
    path_segments: list[tuple[float, float, float, float]],
    doors: list[tuple[str, float, float, float, float]],
    output_path: str,
):
    """Inject <line> elements into Paths and Doors groups."""
    root = tree.getroot()

    paths_grp = root.find(".//svg:g[@id='Paths']", NS)
    if paths_grp is None:
        print("  WARNING: Paths group not found, creating one")
        paths_grp = ET.SubElement(root, "g", {
            "id": "Paths",
            f"{{{INKSCAPE_NS}}}label": "Paths",
            f"{{{INKSCAPE_NS}}}groupmode": "layer",
            "display": "none", "fill": "none",
            "stroke": "#008000", "stroke-width": "3.3",
        })
    for child in list(paths_grp):
        paths_grp.remove(child)

    for i, (x1, y1, x2, y2) in enumerate(path_segments):
        ET.SubElement(paths_grp, f"{{{SVG_NS}}}line", {
            "id": f"p{i + 1}",
            "x1": str(x1), "y1": str(y1),
            "x2": str(x2), "y2": str(y2),
            "display": "inline",
        })

    doors_grp = root.find(".//svg:g[@id='Doors']", NS)
    if doors_grp is None:
        print("  WARNING: Doors group not found, creating one")
        doors_grp = ET.SubElement(root, "g", {
            "id": "Doors",
            f"{{{INKSCAPE_NS}}}label": "Doors",
            f"{{{INKSCAPE_NS}}}groupmode": "layer",
            "display": "none", "fill": "none",
            "stroke": "#ff0000", "stroke-width": "3.3",
        })
    for child in list(doors_grp):
        doors_grp.remove(child)

    for door_id, x1, y1, x2, y2 in doors:
        ET.SubElement(doors_grp, f"{{{SVG_NS}}}line", {
            "id": door_id,
            "x1": str(x1), "y1": str(y1),
            "x2": str(x2), "y2": str(y2),
            "display": "inline",
        })

    ET.register_namespace("", SVG_NS)
    ET.register_namespace("inkscape", INKSCAPE_NS)
    ET.register_namespace("sodipodi", "http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd")

    ET.indent(tree, space="    ")
    with open(output_path, "wb") as f:
        tree.write(f, encoding="UTF-8", xml_declaration=True)
    print(f"  SVG written to: {output_path}")


# ── Main Pipeline ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Auto-generate corridor paths and booth doors for inMapper SVGs (v2 - Voronoi)"
    )
    ap.add_argument("svg", help="Input SVG file path")
    ap.add_argument("-o", "--output", required=True, help="Output SVG file path")
    ap.add_argument("--sample-spacing", type=float, default=3.0,
                    help="Booth edge sample spacing (SVG units, default: 3)")
    ap.add_argument("--boundary-spacing", type=float, default=10.0,
                    help="Boundary edge sample spacing (SVG units, default: 10)")
    ap.add_argument("--min-corridor", type=float, default=5.0,
                    help="Minimum corridor width to place doors (default: 5)")
    ap.add_argument("--door-len", type=float, default=8.0,
                    help="Door segment length (default: 8)")
    ap.add_argument("--simplify", type=float, default=2.5,
                    help="Path simplification tolerance (default: 2.5)")
    ap.add_argument("--min-deadend", type=float, default=12.0,
                    help="Minimum dead-end branch length to keep (default: 12)")
    ap.add_argument("--angle-snap", type=float, default=10.0,
                    help="Angle threshold for H/V snapping in degrees (default: 10)")
    ap.add_argument("--buffer", type=float, default=3.0,
                    help="Booth buffer distance for boundary fallback (default: 3)")

    args = ap.parse_args()

    # Step 1: Parse SVG
    print("Step 1: Parsing SVG...")
    tree, booths, walking, building, svg_w, svg_h = parse_svg(args.svg)
    print(f"  SVG size: {svg_w} x {svg_h}")
    print(f"  Booths: {len(booths)}")
    print(f"  Walking area: {'found' if walking else 'not found'}")
    print(f"  Building: {'found' if building else 'not found'}")

    if not booths:
        print("ERROR: No booth polygons found!")
        return

    # Step 2: Determine walkable boundary
    # Building defines the actual hall floor; Walking may extend beyond it.
    print("\nStep 2: Computing walkable boundary...")
    if building is not None and not building.is_empty:
        boundary = building
        if walking is not None and not walking.is_empty:
            clipped = walking.intersection(building)
            if not clipped.is_empty and clipped.area > boundary.area * 0.5:
                boundary = clipped
            print(f"  Using Building as boundary (Walking clipped to Building)")
        else:
            print(f"  Using Building as boundary")
    elif walking is not None and not walking.is_empty:
        boundary = walking
        print(f"  Using Walking as boundary (no Building found)")
    else:
        all_pts = []
        for b in booths:
            all_pts.extend(b.polygon.exterior.coords)
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        margin = max(svg_w, svg_h) * 0.02
        boundary = box(min(xs) - margin, min(ys) - margin,
                       max(xs) + margin, max(ys) + margin)
        print(f"  Using bounding box as boundary")

    print(f"  Boundary area: {boundary.area:.0f} sq units")

    # Step 3: Voronoi corridor extraction
    print("\nStep 3: Computing Voronoi corridors...")
    raw_segments = compute_voronoi_corridors(
        booths, boundary,
        booth_spacing=args.sample_spacing,
        boundary_spacing=args.boundary_spacing,
        min_corridor_width=args.min_corridor,
    )

    if not raw_segments:
        print("ERROR: No corridor segments found!")
        return

    # Step 4: Graph cleanup
    print("\nStep 4: Cleaning path graph...")
    path_segments = process_paths(
        raw_segments, booths,
        min_deadend=args.min_deadend,
        simplify_tol=args.simplify,
        angle_thr=args.angle_snap,
    )

    # Step 5: Generate doors (with T-junction splitting)
    print("\nStep 5: Generating doors with T-junctions...")
    path_segments, doors = generate_doors(
        booths, path_segments,
        door_len=args.door_len,
        min_corridor_width=args.min_corridor,
    )
    print(f"  Doors: {len(doors)}")
    print(f"  Paths after door splits: {len(path_segments)}")

    # Step 6: Write SVG
    print("\nStep 6: Writing output SVG...")
    inject_paths_and_doors(tree, path_segments, doors, args.output)

    print(f"\nDone! {len(path_segments)} paths + {len(doors)} doors generated.")


if __name__ == "__main__":
    main()
