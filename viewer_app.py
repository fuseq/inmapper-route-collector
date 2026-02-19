"""
Route Description Collector - Flask Web App
Dinamik oda seçimi ile rota hesaplama ve veri toplama aracı
"""
import sys
import os
os.environ['PYTHONUNBUFFERED'] = '1'

from flask import Flask, jsonify, request, send_file, render_template_string
from flask_cors import CORS
import json
import os
import xml.etree.ElementTree as ET
import urllib.request

from helpers.dijkstra import dijkstra_connections
from helpers.extract_xml import build_graph, get_room_areas
from helpers.path_analysis import NavigationGraphCleaner
from helpers.route_visualizer import RouteVisualizer
from helpers.direction_calculator import calculate_initial_direction
from helpers.batch_route_generator import generate_alternative_routes_for_room_pair

app = Flask(__name__)
CORS(app)

# ─── Global state (venue verileri) ───
_state = {
    'graphs': [],
    'floor_areas': [],
    'floor_names': [],
    'svg_paths': {},
    'venue': None
}


def init_venue(venue="zorlu"):
    """Venue verilerini yükle ve cache'le"""
    if _state['venue'] == venue:
        return  # Zaten yüklü

    if venue == "zorlu":
        svg_paths = {
            "Kat 0": "files/floors/0.svg",
            "Kat -1": "files/floors/-1.svg",
            "Kat -2": "files/floors/-2.svg",
            "Kat -3": "files/floors/-3.svg"
        }
        portals_path = "files/supportive/portals.json"
        floor_names = ["Kat 0", "Kat -1", "Kat -2", "Kat -3"]
    else:
        svg_paths = {
            "Kat 0": "files/floors/ankarasehir/0.svg",
            "Kat -1": "files/floors/ankarasehir/-1.svg",
            "Kat -2": "files/floors/ankarasehir/-2.svg"
        }
        portals_path = "files/supportive/portals_ankarasehir.json"
        floor_names = ["Kat 0", "Kat -1", "Kat -2"]

    floor_areas = []
    for floor_name in floor_names:
        svg_path = svg_paths[floor_name]
        areas = get_room_areas(svg_path)
        floor_areas.append(areas)
        room_count = sum(len(rooms) for rooms in areas.values())
        print(f"  {floor_name}: {room_count} oda")

    graphs = []
    for floor_name in floor_names:
        svg_path = svg_paths[floor_name]
        graph = build_graph(svg_path, portals_path)
        graph.find_intersections()
        graph._nav_cleaner = NavigationGraphCleaner(graph)
        graphs.append(graph)

    _state['graphs'] = graphs
    _state['floor_areas'] = floor_areas
    _state['floor_names'] = floor_names
    _state['svg_paths'] = svg_paths
    _state['venue'] = venue
    print(f"✓ {venue} venue verileri yüklendi")


def _find_room(floor_name, room_id):
    """Verilen kattaki oda bilgisini bul"""
    floor_idx = None
    for i, fn in enumerate(_state['floor_names']):
        if fn == floor_name:
            floor_idx = i
            break
    if floor_idx is None:
        return None, None, None

    areas = _state['floor_areas'][floor_idx]
    for room_type, rooms in areas.items():
        for room in rooms:
            if room['id'] == room_id:
                return {
                    'id': room['id'],
                    'type': room_type,
                    'center': room['center'],
                    'area': room['area']
                }, floor_idx, areas
    return None, None, None


def _generate_route_svg(svg_path, connection_ids, anchor_ids, path_points):
    """Rota SVG içeriğini bellekte oluştur (dosyaya yazmadan)"""
    visualizer = RouteVisualizer(svg_path)
    visualizer.highlight_route_connections(
        connection_ids=connection_ids,
        route_color='#FF0000',
        route_width=4.0
    )
    if anchor_ids:
        visualizer.highlight_anchor_rooms(
            anchor_ids=anchor_ids,
            anchor_color='#FFD700'
        )

    # Yön oku
    if path_points and len(path_points) >= 2:
        pts = [tuple(p) for p in path_points]
        direction = calculate_initial_direction(pts, max_segments=5, min_segment_length=0)
        if direction:
            visualizer.draw_direction_arrow(
                start_point=direction['start_point'],
                direction_vector=direction['direction_vector'],
                arrow_length=150.0,
                arrow_color='#00FFFF',
                arrow_width=8.0,
                label=direction['compass'],
                confidence=direction['confidence'],
                compass=direction['compass']
            )

    # SVG'yi string olarak döndür
    svg_ns = visualizer.namespace['svg']
    ET.register_namespace('', svg_ns)
    return ET.tostring(visualizer.root, encoding='unicode')


# ─── API Endpoints ───

@app.route('/')
def index():
    """Ana sayfa — viewer HTML"""
    return send_file('route_viewer_dynamic.html')


@app.route('/api/rooms')
def get_rooms():
    """Tüm katların oda listesini döndür"""
    result = {}
    allowed_types = {'Shop', 'Food', 'Medical', 'Commercial', 'Social'}

    for i, floor_name in enumerate(_state['floor_names']):
        rooms = []
        areas = _state['floor_areas'][i]
        for room_type, room_list in areas.items():
            if room_type not in allowed_types:
                continue
            for room in room_list:
                rooms.append({
                    'id': room['id'],
                    'type': room_type,
                    'label': f"{room_type} - {room['id']}"
                })
        rooms.sort(key=lambda r: r['id'])
        result[floor_name] = rooms

    return jsonify(result)


@app.route('/api/floor-svg/<floor_name>')
def get_floor_svg(floor_name):
    """Seçili katın ham SVG içeriğini döndür (rota olmadan)"""
    if floor_name not in _state['svg_paths']:
        return jsonify({'error': f'Kat bulunamadı: {floor_name}'}), 404

    svg_path = _state['svg_paths'][floor_name]
    try:
        # RouteVisualizer ile SVG'yi yükle (grupları görünür yapar)
        visualizer = RouteVisualizer(svg_path)
        svg_ns = visualizer.namespace['svg']
        ET.register_namespace('', svg_ns)
        svg_content = ET.tostring(visualizer.root, encoding='unicode')
        return svg_content, 200, {'Content-Type': 'image/svg+xml; charset=utf-8'}
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/route', methods=['POST'])
def calculate_route():
    """Rota hesapla ve SVG + steps döndür"""
    try:
        data = request.get_json(force=True, silent=True)
        print(f"[API] Gelen veri: {data}", flush=True)
        if not data:
            print("[API] HATA: JSON body yok", flush=True)
            return jsonify({'error': 'JSON body gerekli'}), 400
        
        start_floor = data.get('start_floor', 'Kat 0')
        start_id = data.get('start_id')
        end_floor = data.get('end_floor', 'Kat 0')
        end_id = data.get('end_id')

        print(f"[API] Parsed: floor={start_floor}, start={start_id}, end={end_id}", flush=True)

        if not start_id or not end_id:
            print(f"[API] HATA: start_id={start_id}, end_id={end_id}", flush=True)
            return jsonify({'error': 'start_id ve end_id gerekli'}), 400

        if start_id == end_id and start_floor == end_floor:
            return jsonify({'error': 'Başlangıç ve hedef aynı olamaz'}), 400

        print(f"[API] Rota hesaplanıyor: {start_floor}/{start_id} → {end_floor}/{end_id}", flush=True)

        # Odaları bul
        start_room, start_idx, start_areas = _find_room(start_floor, start_id)
        end_room, end_idx, end_areas = _find_room(end_floor, end_id)

        if not start_room:
            print(f"[API] HATA: Başlangıç odası bulunamadı: {start_id}", flush=True)
            return jsonify({'error': f'Başlangıç odası bulunamadı: {start_id}'}), 400
        if not end_room:
            print(f"[API] HATA: Hedef oda bulunamadı: {end_id}", flush=True)
            return jsonify({'error': f'Hedef oda bulunamadı: {end_id}'}), 400

        # Şimdilik sadece aynı kat destekleniyor
        if start_floor != end_floor:
            return jsonify({'error': 'Şimdilik sadece aynı kat rotaları destekleniyor'}), 400

        graph = _state['graphs'][start_idx]

        # Rota hesapla
        alternatives = generate_alternative_routes_for_room_pair(
            start_room=start_room,
            end_room=end_room,
            graph=graph,
            floor_areas=start_areas,
            pixel_to_meter_ratio=0.1
        )

        if not alternatives or 'routes' not in alternatives:
            print(f"[API] HATA: Rota bulunamadı", flush=True)
            return jsonify({'error': 'Rota bulunamadı'}), 400

        # Sadece shortest rotayı al
        route_info = alternatives['routes'].get('shortest')
        if not route_info:
            route_info = list(alternatives['routes'].values())[0]

        # SVG oluştur
        svg_path = _state['svg_paths'][start_floor]
        anchor_ids = []
        if 'turns' in route_info:
            for turn in route_info['turns']:
                if turn.get('anchor'):
                    anchor_ids.append(turn['anchor'][1])

        svg_content = _generate_route_svg(
            svg_path=svg_path,
            connection_ids=route_info['path_connections'],
            anchor_ids=anchor_ids,
            path_points=route_info.get('path_points', [])
        )

        # Steps hazırla
        steps = [
            {
                'step_number': s['step_number'],
                'action': s['action'],
                'distance_meters': s.get('distance_meters', 0),
                'description': s.get('description', ''),
                'landmark': s.get('landmark', None)
            }
            for s in route_info.get('steps', [])
        ]

        print(f"[API] ✓ Rota hazır: {len(steps)} adım, {route_info['summary']['total_distance_meters']:.1f}m", flush=True)

        return jsonify({
            'svg': svg_content,
            'steps': steps,
            'distance': route_info['summary']['total_distance_meters'],
            'turns': route_info['turns_count'],
            'start': f"{start_room['type']} - {start_room['id']}",
            'end': f"{end_room['type']} - {end_room['id']}",
            'floor': start_floor
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── Google Sheets'e veri gönder ───
GOOGLE_SHEETS_URL = os.environ.get('GOOGLE_SHEETS_URL', '')


def _format_step_line(step):
    """Tek bir step'i metrik formatta satıra çevir"""
    num = step.get('step_number', 0)
    action = step.get('action', '').upper()
    dist = step.get('distance_meters', 0)
    landmark = step.get('landmark', '')
    desc = step.get('system_description', '')

    # "3. TURN_LEFT    11.2m [Food - ID017]       | Sola sonra ..."
    action_str = f"{action:<13}"
    dist_str = f"{dist:.1f}m"

    if landmark:
        ref_str = f" [{landmark}]"
    else:
        ref_str = ""

    return f"{num}. {action_str} {dist_str}{ref_str}  |  {desc}"


def _build_sheet_row(data):
    """Frontend verisini tek bir Google Sheets satırına dönüştür"""
    start_room = data.get('start_room', '')
    end_room = data.get('end_room', '')
    floor = data.get('floor', '0')
    venue = data.get('venue', 'zorlu')
    steps = data.get('steps', [])

    # ID: "Kat 0_Food_ID007_to_Kat 0_Shop_ID005"
    start_compact = start_room.replace(' - ', '_').replace(' ', '_')
    end_compact = end_room.replace(' - ', '_').replace(' ', '_')
    row_id = f"Kat {floor}_{start_compact}_to_Kat {floor}_{end_compact}"

    # Metric Steps: her step tam formatlı satır
    metric_lines = []
    for step in steps:
        metric_lines.append(_format_step_line(step))
    metric_text = "\n".join(metric_lines)

    # Human Steps: her step numaralı kullanıcı tarifi
    human_lines = []
    for step in steps:
        num = step.get('step_number', 0)
        human_desc = step.get('human_description', '').strip()
        human_lines.append(f"{num}. {human_desc}")
    human_text = "\n".join(human_lines)

    return {
        'id': row_id,
        'venue': venue,
        'start_room': start_room,
        'end_room': end_room,
        'floor': f"Kat {floor}",
        'metric_steps': metric_text,
        'human_steps': human_text,
        'timestamp': data.get('timestamp', ''),
        'step_count': len(steps)
    }


@app.route('/api/submit', methods=['POST'])
def submit_descriptions():
    """Kullanıcı tariflerini Google Sheets'e gönder"""
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Veriyi sheet formatına çevir
        sheet_row = _build_sheet_row(data)
        print(f"[SUBMIT] ID: {sheet_row['id']}, steps: {sheet_row['step_count']}", flush=True)

        if not GOOGLE_SHEETS_URL:
            print("[SUBMIT] GOOGLE_SHEETS_URL not configured, saving locally", flush=True)
            os.makedirs('submissions', exist_ok=True)
            ts = data.get('timestamp', '').replace(':', '-').replace('T', '_')[:19]
            fname = f"submissions/tarif_{ts}.json"
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(sheet_row, f, ensure_ascii=False, indent=2)
            return jsonify({'status': 'saved_locally', 'file': fname})

        # Google Apps Script'e POST et
        payload = json.dumps(sheet_row).encode('utf-8')
        req = urllib.request.Request(
            GOOGLE_SHEETS_URL,
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode('utf-8'))

        print(f"[SUBMIT] Google Sheets OK: {result}", flush=True)
        return jsonify({'status': 'ok', 'sheet_result': result})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def start_app():
    """Venue verilerini yükle ve sunucuyu başlat"""
    print("\n" + "=" * 60)
    print("  Route Description Collector")
    print("  Initializing venue data...")
    print("=" * 60)

    venue = os.environ.get('VENUE', 'zorlu')
    init_venue(venue)

    port = int(os.environ.get('PORT', 5001))
    print(f"\n" + "=" * 60)
    print(f"  Server starting at http://0.0.0.0:{port}")
    print("=" * 60 + "\n")

    return port


if __name__ == '__main__':
    port = start_app()
    app.run(debug=False, host='0.0.0.0', port=port)

