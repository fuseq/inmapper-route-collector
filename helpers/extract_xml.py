import json
import xml.etree.ElementTree as ET
import os
from typing import List, Tuple, Dict
import re
import math

from helpers.dijkstra import Graph
from helpers.path_analysis import extract_path_points

# Dosya adından layer_id çekme
def get_layer_id_from_filename(svg_file_path):
    return os.path.splitext(os.path.basename(svg_file_path))[0]

# JSON dosyasını yükleme fonksiyonu
def load_portal_statuses(json_file_path):
    with open(json_file_path, 'r') as f:
        return json.load(f)

def path2Line(path):
    line = {}
    d = path.split(' ')
    if d[0] in ['M', 'm']:
        start = d[1].split(',')
        line['x1'] = float(start[0])
        line['y1'] = float(start[1])
    if d[2] == 'H':
        line['x2'] = float(d[3])
        line['y2'] = line['y1']
    elif d[2] == 'h':
        line['x2'] = line['x1'] + float(d[3])
        line['y2'] = line['y1']
    elif d[2] == 'V':
        line['x2'] = line['x1']
        line['y2'] = float(d[3])
    elif d[2] == 'v':
        line['x2'] = line['x1']
        line['y2'] = line['y1'] + float(d[3])
    else:
        end = d[2].split(',')
        if d[0] == 'M':
            line['x2'] = float(end[0])
            line['y2'] = float(end[1])
        elif d[0] == 'm':
            line['x2'] = line['x1'] + float(end[0])
            line['y2'] = line['y1'] + float(end[1])
    return line


def parse_paths(svg_file_path, layer_id):
    tree = ET.parse(svg_file_path)
    root = tree.getroot()

    namespace = {"svg": root.tag.split('}')[0].strip('{')}
    paths_group = root.find(".//svg:g[@id='Paths']", namespace)
    if paths_group is None:
        print("Group with ID 'Paths' not found.")
        return []

    combined_data = []

    for line in paths_group.findall(".//svg:line", namespace):
        line_id = line.get("id")
        x1 = float(line.get("x1"))
        y1 = float(line.get("y1"))
        x2 = float(line.get("x2"))
        y2 = float(line.get("y2"))
        combined_data.append({
            "id": line_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "layer_id": layer_id,
            "status": "On"  # Varsayılan durum
        })

    for path in paths_group.findall(".//svg:path", namespace):
        path_id = path.get("id")
        d = path.get("d")
        try:
            line_data = path2Line(d)
            combined_data.append({
                "id": path_id,
                "x1": line_data['x1'],
                "y1": line_data['y1'],
                "x2": line_data['x2'],
                "y2": line_data['y2'],
                "layer_id": layer_id,
                "status": "On"  # Varsayılan durum
            })
        except Exception as e:
            print(f"Failed to parse 'd' attribute for {path_id}: {e}")
            continue

    return combined_data


def parse_doors(svg_file_path, layer_id):
    tree = ET.parse(svg_file_path)
    root = tree.getroot()

    namespace = {"svg": root.tag.split('}')[0].strip('{')}
    doors_group = root.find(".//svg:g[@id='Doors']", namespace)
    if doors_group is None:
        print("Group with ID 'Doors' not found.")
        return []

    doors_data = []

    # <line> elementlerini işle
    for line in doors_group.findall(".//svg:line", namespace):
        line_id = line.get("id")
        x1 = float(line.get("x1"))
        y1 = float(line.get("y1"))
        x2 = float(line.get("x2"))
        y2 = float(line.get("y2"))
        doors_data.append({
            "id": line_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "layer_id": layer_id,
            "status": "On"  # Varsayılan durum
        })

    # <path> elementlerini de işle (ankarasehir gibi SVG'ler için)
    for path in doors_group.findall(".//svg:path", namespace):
        path_id = path.get("id")
        d = path.get("d")
        try:
            line_data = path2Line(d)
            doors_data.append({
                "id": path_id,
                "x1": line_data['x1'],
                "y1": line_data['y1'],
                "x2": line_data['x2'],
                "y2": line_data['y2'],
                "layer_id": layer_id,
                "status": "On"  # Varsayılan durum
            })
        except Exception as e:
            continue

    return doors_data


def parse_portals(svg_file_path, layer_id, portal_statuses):
    tree = ET.parse(svg_file_path)
    root = tree.getroot()

    namespace = {"svg": root.tag.split('}')[0].strip('{')}
    portals_group = root.find(".//svg:g[@id='Portals']", namespace)
    if portals_group is None:
        print("Group with ID 'Portals' not found.")
        return []

    portals_data = []

    # <line> elementlerini işle
    for line in portals_group.findall(".//svg:line", namespace):
        line_id = line.get("id")
        x1 = float(line.get("x1"))
        y1 = float(line.get("y1"))
        x2 = float(line.get("x2"))
        y2 = float(line.get("y2"))
        # JSON'dan status bilgisini çek
        status = next((p["Status"] for p in portal_statuses if p["id"] == line_id and p["layerId"] == layer_id), "Unknown")
        portals_data.append({
            "id": line_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "layer_id": layer_id,
            "status": status
        })
    
    # <path> elementlerini de işle (ankarasehir gibi SVG'ler için)
    for path_elem in portals_group.findall(".//svg:path", namespace):
        path_id = path_elem.get("id")
        d = path_elem.get("d")
        if d and path_id:
            try:
                # Path verisinden koordinatları çıkar
                coords = parse_path_to_line_coords(d)
                if coords:
                    # JSON'dan status bilgisini çek
                    status = next((p["Status"] for p in portal_statuses if p["id"] == path_id and p["layerId"] == layer_id), "Unknown")
                    portals_data.append({
                        "id": path_id,
                        "x1": coords['x1'],
                        "y1": coords['y1'],
                        "x2": coords['x2'],
                        "y2": coords['y2'],
                        "layer_id": layer_id,
                        "status": status
                    })
            except Exception as e:
                print(f"Portal path işlenirken hata: {path_id} - {e}")
    
    print(f"Parsed Portals: {len(portals_data)} portal bulundu")
    return portals_data


def parse_path_to_line_coords(d):
    """
    Basit path verisini line koordinatlarına çevir.
    Örnek: "m 177.95318,1633.3954 v -5.3546" -> {x1, y1, x2, y2}
    """
    try:
        parts = d.strip().split()
        if len(parts) < 2:
            return None
        
        # Başlangıç noktası
        if parts[0].lower() == 'm':
            start = parts[1].split(',')
            x1 = float(start[0])
            y1 = float(start[1])
        else:
            return None
        
        # Bitiş noktası
        x2, y2 = x1, y1
        
        if len(parts) >= 4:
            cmd = parts[2].lower()
            value = float(parts[3])
            
            if cmd == 'h':  # yatay (relative)
                x2 = x1 + value
            elif cmd == 'v':  # dikey (relative)
                y2 = y1 + value
            elif cmd == 'l':  # line to (relative)
                coords = parts[3].split(',')
                x2 = x1 + float(coords[0])
                y2 = y1 + float(coords[1]) if len(coords) > 1 else y1
        
        return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    except:
        return None


def parse_anchor_points(svg_file_path: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Parse anchor points from the SVG file's 'Rooms' group, specifically from Shop, Food, and Other subgroups.
    Returns a dictionary mapping anchor IDs to their points.
    """
    tree = ET.parse(svg_file_path)
    root = tree.getroot()
    namespace = {"svg": root.tag.split('}')[0].strip('{')}
    
    anchor_points = {}
    rooms_group = root.find(".//svg:g[@id='Rooms']", namespace)
    
    if rooms_group is not None:
        # Look for Shop, Food, and Other groups
        target_groups = ['Shop', 'Food', 'Other']
        for group_id in target_groups:
            group = rooms_group.find(f".//svg:g[@id='{group_id}']", namespace)
            if group is not None:
                for path in group.findall(".//svg:path", namespace):
                    path_id = path.get('id')
                    d_attr = path.get('d')
                    if path_id and d_attr:
                        # Parse the d attribute to extract points
                        points = []
                        commands = d_attr.split()
                        current_point = None
                        i = 0
                        
                        while i < len(commands):
                            cmd = commands[i]
                            if cmd in ['M', 'm', 'L', 'l', 'z', 'Z']:
                                if cmd.lower() == 'z':
                                    if points:  # Close the path by adding the first point
                                        points.append(points[0])
                                    i += 1
                                    continue
                                    
                                try:
                                    x = float(commands[i + 1].split(',')[0])
                                    y = float(commands[i + 1].split(',')[1])
                                    
                                    if cmd.islower():  # Relative coordinates
                                        if current_point:
                                            x += current_point[0]
                                            y += current_point[1]
                                    
                                    current_point = (x, y)
                                    points.append(current_point)
                                    i += 2
                                except (IndexError, ValueError):
                                    i += 1
                            else:
                                i += 1
                        
                        if points:
                            anchor_points[f"{group_id}_{path_id}"] = points
                    
    return anchor_points


def build_graph(svg_file_path, portals_json_path):
    layer_id = os.path.splitext(os.path.basename(svg_file_path))[0]
    graph = Graph(layer_id)

    # Load portal statuses
    portal_statuses = load_portal_statuses(portals_json_path)

    # Parse paths
    paths = parse_paths(svg_file_path, layer_id)
    for path in paths:
        graph.add_connection(
            path['x1'], path['y1'], path['x2'], path['y2'], 
            "path", path['id'], path['layer_id'], path['status']
        )

    # Parse doors
    doors = parse_doors(svg_file_path, layer_id)
    for door in doors:
        graph.add_connection(
            door['x1'], door['y1'], door['x2'], door['y2'],
            "door", door['id'], door['layer_id'], door['status']
        )

    # Parse portals
    portals = parse_portals(svg_file_path, layer_id, portal_statuses)
    for portal in portals:
        graph.add_connection(
            portal['x1'], portal['y1'], portal['x2'], portal['y2'],
            "portal", portal['id'], portal['layer_id'], portal['status']
        )

    # Parse and add anchor points
    anchor_points = parse_anchor_points(svg_file_path)
    for anchor_id, points in anchor_points.items():
        graph.add_anchor_point(anchor_id, points)

    return graph

def parse_path_d(d):
    """Parse SVG path data string into a list of commands and coordinates"""
    tokens = re.findall(r"[a-df-zA-DF-Z]|[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", d)
    commands = []
    i = 0
    while i < len(tokens):
        if tokens[i].isalpha():
            cmd = tokens[i]  # Preserve case (uppercase=absolute, lowercase=relative)
            i += 1
            coords = []
            while i < len(tokens) and not tokens[i].isalpha():
                coords.append(float(tokens[i]))
                i += 1
            commands.append((cmd, coords))
        else:
            i += 1
    return commands

def parse_path_to_absolute_coords(d):
    """Convert SVG path data to absolute coordinates"""
    commands = parse_path_d(d)
    points = []
    current_x, current_y = 0, 0
    start_x, start_y = 0, 0  # Subpath start point for 'z' command
    
    for cmd, coords in commands:
        # Move commands
        if cmd == 'm':  # Relative move
            if not points:  # First move is treated as absolute
                current_x, current_y = coords[0], coords[1]
            else:
                current_x += coords[0]
                current_y += coords[1]
            start_x, start_y = current_x, current_y
            points.append((current_x, current_y))
            # Subsequent coordinates are relative line-to
            for i in range(2, len(coords), 2):
                current_x += coords[i]
                current_y += coords[i + 1]
                points.append((current_x, current_y))
                
        elif cmd == 'M':  # Absolute move
            current_x, current_y = coords[0], coords[1]
            start_x, start_y = current_x, current_y
            points.append((current_x, current_y))
            # Subsequent coordinates are absolute line-to
            for i in range(2, len(coords), 2):
                current_x, current_y = coords[i], coords[i + 1]
                points.append((current_x, current_y))
                
        # Line commands
        elif cmd == 'l':  # Relative line
            for i in range(0, len(coords), 2):
                current_x += coords[i]
                current_y += coords[i + 1]
                points.append((current_x, current_y))
                
        elif cmd == 'L':  # Absolute line
            for i in range(0, len(coords), 2):
                current_x, current_y = coords[i], coords[i + 1]
                points.append((current_x, current_y))
                
        # Horizontal line commands
        elif cmd == 'h':  # Relative horizontal
            for val in coords:
                current_x += val
                points.append((current_x, current_y))
            
        elif cmd == 'H':  # Absolute horizontal
            for val in coords:
                current_x = val
                points.append((current_x, current_y))
                
        # Vertical line commands
        elif cmd == 'v':  # Relative vertical
            for val in coords:
                current_y += val
                points.append((current_x, current_y))
            
        elif cmd == 'V':  # Absolute vertical
            for val in coords:
                current_y = val
                points.append((current_x, current_y))
            
        # Close path
        elif cmd in ('z', 'Z'):
            if points:
                points.append((start_x, start_y))
    
    return points

def calculate_area(coords):
    """Calculate area using the shoelace formula with absolute coordinates"""
    if len(coords) < 3:  # Need at least 3 points to form an area
        return 0
        
    area = 0
    n = len(coords)
    for i in range(n - 1):  # n-1 because the last point is the same as first
        x1, y1 = coords[i]
        x2, y2 = coords[i + 1]
        area += x1 * y2 - x2 * y1
    
    return abs(area) / 2

def calculate_center_point(coords):
    """Calculate the center point of a room from its coordinates"""
    if not coords:
        return None
        
    x_coords = [p[0] for p in coords]
    y_coords = [p[1] for p in coords]
    
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    
    return (center_x, center_y)

def get_room_areas(svg_file_path):
    """Extract and calculate areas for all room types in the SVG file"""
    tree = ET.parse(svg_file_path)
    root = tree.getroot()
    
    # Define namespace
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    
    # Ayrıca inkscape namespace'i de kontrol et
    inkscape_ns = {'inkscape': 'http://www.inkscape.org/namespaces/inkscape'}
    
    # Find the Rooms group
    rooms_group = root.find(".//svg:g[@id='Rooms']", ns)
    if rooms_group is None:
        # inkscape:label ile de dene
        for g in root.findall(".//svg:g", ns):
            label = g.get('{http://www.inkscape.org/namespaces/inkscape}label')
            if label == 'Rooms':
                rooms_group = g
                break
    
    if rooms_group is None:
        print(f"'Rooms' group not found in {svg_file_path}")
        return {}
    
    room_areas = {}
    
    # Rooms grubu altındaki TÜM alt grupları bul (dinamik)
    for child_group in rooms_group.findall("./svg:g", ns):
        # Grup ID'sini veya inkscape:label'ı al
        room_type = child_group.get('id')
        if not room_type:
            room_type = child_group.get('{http://www.inkscape.org/namespaces/inkscape}label')
        
        if not room_type:
            continue
        
        room_areas[room_type] = []
        
        for path in child_group.findall("./svg:path", ns):
            d = path.get('d')
            if d:
                try:
                    # Convert path to absolute coordinates
                    coords = parse_path_to_absolute_coords(d)
                    
                    # Calculate area and center point
                    area = calculate_area(coords)
                    center = calculate_center_point(coords)
                    
                    room_id = path.get('id', 'unknown')
                    room_areas[room_type].append({
                        'id': room_id,
                        'area': area,
                        'center': center,
                        'coordinates': coords  # Polygon koordinatlari
                    })
                    
                except Exception as e:
                    room_id = path.get('id', 'unknown')
                    print(f"Error processing room {room_id}: {str(e)}")
    
    # Bulunan oda tiplerini özet olarak yazdır
    if room_areas:
        total_rooms = sum(len(rooms) for rooms in room_areas.values())
        print(f"  Toplam {total_rooms} oda bulundu: {list(room_areas.keys())}")
    
    return room_areas