"""
SVG Viewer Generator - İnteraktif SVG görüntüleyici + Veri Toplama Aracı
Rota görselleştirme ve adım bazlı insan tarifi toplama
"""
import os
import json
from typing import List, Dict, Optional
import webbrowser


def generate_interactive_viewer(svg_files: List[Dict[str, str]], 
                                output_html: str = "route_viewer.html",
                                auto_open: bool = True,
                                route_steps: Optional[Dict[str, List[Dict]]] = None,
                                route_metadata: Optional[Dict] = None) -> str:
    """
    İnteraktif SVG görüntüleyici + veri toplama HTML sayfası oluşturur
    
    Args:
        svg_files: SVG dosya bilgileri listesi
        output_html: Çıktı HTML dosyası adı
        auto_open: Otomatik olarak tarayıcıda açılsın mı
        route_steps: Rota adımları {route_type: [step_dict, ...]}
                     Her step_dict: {step_number, action, distance_meters, description, landmark}
        route_metadata: Rota meta bilgileri {start_room, end_room, venue, ...}
    
    Returns:
        Oluşturulan HTML dosyasının yolu
    """
    
    # SVG dosyalarını oku ve organize et
    svg_contents = []
    floors_dict = {}
    route_types = set()
    
    for svg_file in svg_files:
        try:
            with open(svg_file['path'], 'r', encoding='utf-8') as f:
                content = f.read()
                if content.startswith('<?xml'):
                    content = content[content.index('?>') + 2:].strip()
                
                floor = svg_file.get('floor', 'Kat 0')
                route_type = svg_file.get('route_type', 'shortest')
                name = svg_file.get('name', 'Rota')
                distance = svg_file.get('distance', 0)
                turns = svg_file.get('turns', 0)
                
                if floor not in floors_dict:
                    floors_dict[floor] = len(floors_dict)
                
                route_types.add(route_type)
                
                svg_contents.append({
                    'name': name,
                    'floor': floor,
                    'route_type': route_type,
                    'distance': distance,
                    'turns': turns,
                    'content': content
                })
        except Exception as e:
            print(f"Uyarı: {svg_file['path']} okunamadı: {e}")
    
    if not svg_contents:
        print("Hata: Hiçbir SVG dosyası okunamadı")
        return None
    
    floors = sorted(floors_dict.keys(), key=lambda x: floors_dict[x])
    route_type_order = {'shortest': 0, 'least_turns': 1}
    route_types = sorted(list(route_types), key=lambda x: route_type_order.get(x, 999))
    
    has_multiple_floors = len(floors) > 1
    has_multiple_routes = len(route_types) > 1
    has_steps = route_steps is not None and len(route_steps) > 0
    
    # Route steps JSON
    steps_json = json.dumps(route_steps or {}, ensure_ascii=False)
    metadata_json = json.dumps(route_metadata or {}, ensure_ascii=False)
    
    # HTML oluştur
    html_content = _build_html(
        svg_contents=svg_contents,
        floors=floors,
        route_types=route_types,
        has_multiple_floors=has_multiple_floors,
        has_multiple_routes=has_multiple_routes,
        has_steps=has_steps,
        steps_json=steps_json,
        metadata_json=metadata_json
    )
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ İnteraktif görüntüleyici oluşturuldu: {output_html}")
    
    if auto_open:
        abs_path = os.path.abspath(output_html)
        webbrowser.open('file://' + abs_path)
        print(f"✓ Görüntüleyici tarayıcıda açıldı")
    
    return output_html


def _build_html(svg_contents, floors, route_types, has_multiple_floors, 
                has_multiple_routes, has_steps, steps_json, metadata_json):
    """HTML içeriğini oluşturur"""
    
    # SVG data JS array
    svg_data_items = []
    for i, svg in enumerate(svg_contents):
        svg_data_items.append(
            f'{{ floor: "{svg["floor"]}", routeType: "{svg["route_type"]}", '
            f'name: "{svg["name"]}", distance: {svg["distance"]:.1f}, '
            f'turns: {svg["turns"]}, id: "svg-{i}" }}'
        )
    svg_data_js = ",\n            ".join(svg_data_items)
    
    floors_js = json.dumps(floors, ensure_ascii=False)
    route_types_js = json.dumps(route_types, ensure_ascii=False)
    
    # SVG wrapper divs
    svg_wrappers = ""
    for i, svg in enumerate(svg_contents):
        active = " active" if i == 0 else ""
        svg_wrappers += f'''
                <div class="svg-wrapper{active}" id="svg-{i}" 
                     data-floor="{svg['floor']}" data-route-type="{svg['route_type']}">
                    {svg['content']}
                </div>'''
    
    html = f'''<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Route Description Collector</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f2f5;
            height: 100vh;
            overflow: hidden;
        }}
        
        .app-container {{
            display: flex;
            flex-direction: column;
            height: 100vh;
        }}
        
        /* ─── HEADER ─── */
        .header {{
            background: white;
            padding: 12px 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 100;
            flex-shrink: 0;
        }}
        
        .title {{
            font-size: 20px;
            font-weight: 700;
            color: #1a1a2e;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .title-badge {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .route-type-selector {{
            display: flex;
            gap: 8px;
            background: #f0f2f5;
            padding: 4px;
            border-radius: 20px;
        }}
        
        .route-type-btn {{
            background: transparent;
            border: none;
            padding: 8px 18px;
            border-radius: 16px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.25s ease;
            color: #666;
        }}
        
        .route-type-btn.active {{
            background: white;
            color: #667eea;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .route-type-btn:hover:not(.active) {{
            color: #667eea;
        }}
        
        .header-controls {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        
        .floor-badge {{
            background: #667eea;
            color: white;
            padding: 6px 14px;
            border-radius: 14px;
            font-size: 13px;
            font-weight: 600;
        }}
        
        .btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.2s ease;
        }}
        
        .btn:hover {{ background: #5568d3; }}
        .btn:disabled {{ background: #ccc; cursor: not-allowed; }}
        .btn-secondary {{ background: #6c757d; }}
        .btn-secondary:hover {{ background: #5a6268; }}
        
        /* ─── MAIN CONTENT ─── */
        .main-content {{
            flex: 1;
            display: flex;
            overflow: hidden;
            gap: 12px;
            padding: 12px;
        }}
        
        /* ─── MAP PANEL (LEFT) ─── */
        .map-panel {{
            flex: {"1 1 58%" if has_steps else "1"};
            position: relative;
            background: white;
            min-width: 0;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
            overflow: hidden;
        }}
        
        .svg-container {{
            width: 100%;
            height: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: grab;
            position: relative;
        }}
        
        .svg-container.grabbing {{ cursor: grabbing; }}
        
        .svg-wrapper {{
            display: none;
            transform-origin: 0 0;
        }}
        
        .svg-wrapper.active {{ display: block; }}
        
        .map-info {{
            position: absolute;
            top: 12px;
            left: 12px;
            background: rgba(255,255,255,0.92);
            padding: 10px 14px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            font-size: 13px;
            color: #555;
            z-index: 10;
        }}
        
        .map-info-row {{
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 4px;
        }}
        
        .map-info-row:last-child {{ margin-bottom: 0; }}
        
        .zoom-controls {{
            position: absolute;
            bottom: 16px;
            right: 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            z-index: 10;
        }}
        
        .zoom-btn {{
            width: 40px;
            height: 40px;
            background: white;
            border: 2px solid #667eea;
            border-radius: 50%;
            cursor: pointer;
            font-size: 20px;
            font-weight: bold;
            color: #667eea;
            display: flex;
            justify-content: center;
            align-items: center;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .zoom-btn:hover {{
            background: #667eea;
            color: white;
        }}
        
        /* ─── STEPS PANEL (RIGHT) ─── */
        .steps-panel {{
            flex: 0 0 40%;
            display: flex;
            flex-direction: column;
            min-width: 0;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.06);
            overflow: hidden;
        }}
        
        .steps-header {{
            padding: 16px 20px;
            border-bottom: 1px solid #eee;
            flex-shrink: 0;
        }}
        
        .steps-title {{
            font-size: 16px;
            font-weight: 700;
            color: #1a1a2e;
            margin-bottom: 4px;
        }}
        
        .steps-subtitle {{
            font-size: 12px;
            color: #888;
        }}
        
        .progress-bar {{
            height: 4px;
            background: #e9ecef;
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 2px;
            transition: width 0.3s ease;
            width: 0%;
        }}
        
        .steps-list {{
            flex: 1;
            overflow-y: auto;
            padding: 12px 16px 12px 16px;
            scrollbar-gutter: stable;
        }}
        
        .steps-list::-webkit-scrollbar {{
            width: 6px;
        }}
        
        .steps-list::-webkit-scrollbar-track {{
            background: transparent;
        }}
        
        .steps-list::-webkit-scrollbar-thumb {{
            background: #ddd;
            border-radius: 3px;
        }}
        
        .step-card {{
            border: 2px solid #e9ecef;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 10px;
            transition: all 0.2s ease;
        }}
        
        .step-card:hover {{
            border-color: #667eea40;
        }}
        
        .step-card.filled {{
            border-color: #28a74560;
            background: #f8fff8;
        }}
        
        .step-card-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        
        .step-number {{
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
            font-weight: 700;
            color: white;
            flex-shrink: 0;
        }}
        
        .step-number.start {{ background: #667eea; }}
        .step-number.turn, .step-number.turn_left, .step-number.turn_right {{ background: #e74c3c; }}
        .step-number.veer {{ background: #f39c12; }}
        .step-number.pass, .step-number.pass_by {{ background: #3498db; }}
        .step-number.arrive {{ background: #27ae60; }}
        .step-number.floor_change {{ background: #9b59b6; }}
        
        .step-action {{
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            padding: 2px 8px;
            border-radius: 4px;
            color: white;
        }}
        
        .step-action.start {{ background: #667eea; }}
        .step-action.turn_left, .step-action.turn_right, .step-action.turn {{ background: #e74c3c; }}
        .step-action.veer {{ background: #f39c12; }}
        .step-action.pass_by, .step-action.pass {{ background: #3498db; }}
        .step-action.arrive {{ background: #27ae60; }}
        .step-action.floor_change {{ background: #9b59b6; }}
        
        .step-distance {{
            font-size: 12px;
            color: #888;
            margin-left: auto;
            font-weight: 600;
        }}
        
        .step-landmark {{
            font-size: 12px;
            color: #667eea;
            margin-left: 4px;
            font-weight: 500;
        }}
        
        .step-system-desc {{
            background: #f8f9fa;
            border-left: 3px solid #667eea;
            padding: 8px 12px;
            margin-bottom: 8px;
            border-radius: 0 6px 6px 0;
            font-size: 13px;
            color: #555;
            line-height: 1.5;
            font-family: 'Consolas', 'Courier New', monospace;
            white-space: pre-wrap;
        }}
        
        .step-system-label {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #999;
            margin-bottom: 2px;
            font-weight: 600;
        }}
        
        .step-input-label {{
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #667eea;
            margin-bottom: 4px;
            font-weight: 700;
        }}
        
        .step-textarea {{
            width: 100%;
            min-height: 60px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px 12px;
            font-size: 14px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.2s ease;
            line-height: 1.5;
            color: #333;
        }}
        
        .step-textarea:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15);
        }}
        
        .step-textarea.has-text {{
            border-color: #28a745;
        }}
        
        .step-textarea::placeholder {{
            color: #bbb;
            font-size: 13px;
        }}
        
        /* ─── FOOTER ─── */
        .steps-footer {{
            padding: 14px 20px;
            border-top: 1px solid #eee;
            display: flex;
            flex-direction: column;
            align-items: center;
            flex-shrink: 0;
            gap: 10px;
        }}
        
        .steps-counter {{
            font-size: 13px;
            color: #888;
            font-weight: 500;
        }}
        
        .steps-counter strong {{
            color: #667eea;
        }}
        
        .submit-btn {{
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 15px;
            font-weight: 700;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            width: 100%;
            box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3);
        }}
        
        .submit-btn:hover:not(:disabled) {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(40, 167, 69, 0.4);
        }}
        
        .submit-btn:disabled {{
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }}
        
        /* ─── TOAST ─── */
        .toast {{
            position: fixed;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%) translateY(100px);
            background: #333;
            color: white;
            padding: 14px 28px;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 500;
            z-index: 1000;
            transition: transform 0.3s ease;
            box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        }}
        
        .toast.show {{
            transform: translateX(-50%) translateY(0);
        }}
        
        .toast.success {{ background: #28a745; }}
        .toast.error {{ background: #dc3545; }}
    </style>
</head>
<body>
    <div class="app-container">
        <!-- HEADER -->
        <div class="header">
            <div class="title">
                Route Description Collector
                <span class="floor-badge" id="floorName">{floors[0]}</span>
            </div>
            
            <div class="route-type-selector" id="routeTypeSelector" 
                 style="display: none;">
                <button class="route-type-btn active" onclick="selectRouteType('shortest')">
                    Shortest
                </button>
                <button class="route-type-btn" onclick="selectRouteType('least_turns')">
                    Least Turns
                </button>
            </div>
            
            <div class="header-controls">
                <button class="btn btn-secondary" id="prevBtn" onclick="previousFloor()" 
                        style="display:{"inline-flex" if has_multiple_floors else "none"}">
                    ← Önceki Kat
                </button>
                <button class="btn btn-secondary" id="nextBtn" onclick="nextFloor()" 
                        style="display:{"inline-flex" if has_multiple_floors else "none"}">
                    Sonraki Kat →
                </button>
                <button class="btn btn-secondary" onclick="resetView()">Reset</button>
            </div>
        </div>
        
        <!-- MAIN -->
        <div class="main-content">
            <!-- MAP -->
            <div class="map-panel">
                <div class="map-info">
                    <div class="map-info-row">Zoom: <strong id="zoomLevel">100%</strong></div>
                    <div class="map-info-row">Distance: <strong id="routeDistance">0</strong> m</div>
                    <div class="map-info-row">Turns: <strong id="routeTurns">0</strong></div>
                </div>
                
                <div class="svg-container" id="svgContainer">
                    {svg_wrappers}
                </div>
                
                <div class="zoom-controls">
                    <button class="zoom-btn" onclick="zoomIn()">+</button>
                    <button class="zoom-btn" onclick="zoomOut()">−</button>
                </div>
            </div>
            
            <!-- STEPS PANEL -->
            {"" if not has_steps else '''
            <div class="steps-panel">
                <div class="steps-header">
                    <div class="steps-title">Step Descriptions</div>
                    <div class="steps-subtitle">Write your own description for each step</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill"></div>
                    </div>
                </div>
                
                <div class="steps-list" id="stepsList">
                    <!-- Steps will be injected by JS -->
                </div>
                
                <div class="steps-footer">
                    <div class="steps-counter">
                        <strong id="filledCount">0</strong> / <span id="totalCount">0</span> completed
                    </div>
                    <button class="submit-btn" id="submitBtn" disabled onclick="submitDescriptions()">
                        Submit
                    </button>
                </div>
            </div>
            '''}
        </div>
    </div>
    
    <div class="toast" id="toast"></div>
    
    <script>
        // ─── DATA ───
        const svgData = [
            {svg_data_js}
        ];
        
        const floors = {floors_js};
        const routeTypes = {route_types_js};
        const hasMultipleFloors = {str(has_multiple_floors).lower()};
        const hasMultipleRoutes = {str(has_multiple_routes).lower()};
        const hasSteps = {str(has_steps).lower()};
        
        const routeSteps = {steps_json};
        const routeMetadata = {metadata_json};
        
        let currentFloor = floors[0];
        let currentRouteType = routeTypes[0];
        let scale = 1;
        let translateX = 0;
        let translateY = 0;
        let isDragging = false;
        let startX = 0;
        let startY = 0;
        
        // Her rota tipi için kullanıcı girişlerini sakla
        const userDescriptions = {{}};
        
        // ─── INIT ───
        initializeViewer();
        
        function initializeViewer() {{
            updateDisplay();
            if (hasSteps) {{
                renderSteps();
            }}
        }}
        
        // ─── MAP DISPLAY ───
        function updateDisplay() {{
            document.querySelectorAll('.svg-wrapper').forEach(w => w.classList.remove('active'));
            
            const activeSvg = svgData.find(svg => 
                svg.floor === currentFloor && svg.routeType === currentRouteType
            );
            
            if (activeSvg) {{
                document.getElementById(activeSvg.id).classList.add('active');
                document.getElementById('floorName').textContent = currentFloor;
                document.getElementById('routeDistance').textContent = activeSvg.distance.toFixed(1);
                document.getElementById('routeTurns').textContent = activeSvg.turns;
                
                if (hasMultipleFloors) {{
                    const idx = floors.indexOf(currentFloor);
                    document.getElementById('prevBtn').disabled = idx === 0;
                    document.getElementById('nextBtn').disabled = idx === floors.length - 1;
                }}
            }}
            
            resetView();
        }}
        
        function selectRouteType(routeType) {{
            // Mevcut girişleri kaydet
            if (hasSteps) saveCurrentInputs();
            
            currentRouteType = routeType;
            document.querySelectorAll('.route-type-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            updateDisplay();
            if (hasSteps) renderSteps();
        }}
        
        function previousFloor() {{
            const idx = floors.indexOf(currentFloor);
            if (idx > 0) {{ currentFloor = floors[idx - 1]; updateDisplay(); }}
        }}
        
        function nextFloor() {{
            const idx = floors.indexOf(currentFloor);
            if (idx < floors.length - 1) {{ currentFloor = floors[idx + 1]; updateDisplay(); }}
        }}
        
        // ─── ZOOM & PAN ───
        function applyTransform() {{
            const activeSvg = svgData.find(svg => 
                svg.floor === currentFloor && svg.routeType === currentRouteType
            );
            if (activeSvg) {{
                const el = document.getElementById(activeSvg.id);
                el.style.transform = `translate(${{translateX}}px, ${{translateY}}px) scale(${{scale}})`;
                document.getElementById('zoomLevel').textContent = Math.round(scale * 100) + '%';
            }}
        }}
        
        function zoomAtPoint(factor, clientX, clientY) {{
            const container = document.getElementById('svgContainer');
            const rect = container.getBoundingClientRect();
            
            // Mouse pozisyonu container içinde
            const mouseX = clientX - rect.left;
            const mouseY = clientY - rect.top;
            
            const newScale = Math.min(Math.max(scale * factor, 0.1), 5);
            
            // Mouse altındaki nokta sabit kalacak şekilde translate hesapla
            translateX = mouseX - (mouseX - translateX) * (newScale / scale);
            translateY = mouseY - (mouseY - translateY) * (newScale / scale);
            scale = newScale;
            applyTransform();
        }}
        
        function zoomIn() {{ zoomAtCenter(1.2); }}
        function zoomOut() {{ zoomAtCenter(1 / 1.2); }}
        function zoomAtCenter(factor) {{
            const container = document.getElementById('svgContainer');
            const rect = container.getBoundingClientRect();
            zoomAtPoint(factor, rect.left + rect.width / 2, rect.top + rect.height / 2);
        }}
        function resetView() {{ scale = 1; translateX = 0; translateY = 0; applyTransform(); }}
        
        document.getElementById('svgContainer').addEventListener('wheel', (e) => {{
            e.preventDefault();
            const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
            zoomAtPoint(factor, e.clientX, e.clientY);
        }});
        
        const container = document.getElementById('svgContainer');
        container.addEventListener('mousedown', (e) => {{
            isDragging = true;
            startX = e.clientX - translateX;
            startY = e.clientY - translateY;
            container.classList.add('grabbing');
        }});
        document.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                translateX = e.clientX - startX;
                translateY = e.clientY - startY;
                applyTransform();
            }}
        }});
        document.addEventListener('mouseup', () => {{
            isDragging = false;
            container.classList.remove('grabbing');
        }});
        
        document.addEventListener('keydown', (e) => {{
            if (e.target.tagName === 'TEXTAREA') return; // textarea'dayken kısayolları engelle
            if (e.key === '+' || e.key === '=') zoomIn();
            else if (e.key === '-') zoomOut();
            else if (e.key === '0') resetView();
        }});
        
        // ─── STEPS RENDERING ───
        function getActionClass(action) {{
            const a = action.toLowerCase();
            if (a === 'start') return 'start';
            if (a.includes('turn')) return 'turn';
            if (a === 'veer') return 'veer';
            if (a === 'pass_by') return 'pass_by';
            if (a === 'arrive') return 'arrive';
            if (a === 'floor_change') return 'floor_change';
            return 'start';
        }}
        
        function getActionLabel(action) {{
            const labels = {{
                'START': 'START',
                'TURN_LEFT': 'TURN LEFT',
                'TURN_RIGHT': 'TURN RIGHT',
                'VEER': 'VEER',
                'PASS_BY': 'PASS BY',
                'ARRIVE': 'ARRIVE',
                'FLOOR_CHANGE': 'FLOOR CHANGE'
            }};
            return labels[action] || action;
        }}
        
        function renderSteps() {{
            const stepsEl = document.getElementById('stepsList');
            if (!stepsEl) return;
            
            const steps = routeSteps[currentRouteType] || [];
            if (steps.length === 0) {{
                stepsEl.innerHTML = '<div style="text-align:center;color:#888;padding:40px;">No steps available for this route.</div>';
                return;
            }}
            
            // Kayıtlı girişleri al
            const saved = userDescriptions[currentRouteType] || {{}};
            
            let html = '';
            steps.forEach((step, i) => {{
                const actionClass = getActionClass(step.action);
                const savedText = saved[step.step_number] || '';
                const filledClass = savedText.trim() ? ' filled' : '';
                const textClass = savedText.trim() ? ' has-text' : '';
                
                const distVal = step.distance_meters != null ? step.distance_meters : 0;
                const distStr = distVal.toFixed(1) + 'm';
                const landmark = step.landmark ? step.landmark : '';
                const landmarkBracket = landmark ? `[${{landmark}}]` : '';
                
                // Tam format: "1. START          9.9m [Food - ID007]       | Food - ID007 noktasından çıkıp düz ilerleyin"
                const actionPad = step.action.padEnd(12);
                const distPad = distStr.padStart(5);
                const landmarkPad = landmarkBracket.padEnd(20);
                const fullSystemLine = `${{step.step_number}}. ${{actionPad}} ${{distPad}} ${{landmarkPad}} | ${{step.description}}`;
                
                html += `
                <div class="step-card${{filledClass}}" id="card-${{step.step_number}}">
                    <div class="step-card-header">
                        <div class="step-number ${{actionClass}}">${{step.step_number}}</div>
                        <span class="step-action ${{actionClass}}">${{getActionLabel(step.action)}}</span>
                        ${{landmark ? `<span class="step-landmark">${{landmark}}</span>` : ''}}
                        ${{distStr !== '0.0m' ? `<span class="step-distance">${{distStr}}</span>` : ''}}
                    </div>
                    <div class="step-system-label">System</div>
                    <div class="step-system-desc">${{fullSystemLine}}</div>
                    <div class="step-input-label">Your Description</div>
                    <textarea 
                        class="step-textarea${{textClass}}" 
                        id="input-${{step.step_number}}" 
                        data-step="${{step.step_number}}"
                        placeholder="Write your description for this step..."
                        oninput="onStepInput(this)"
                    >${{savedText}}</textarea>
                </div>`;
            }});
            
            stepsEl.innerHTML = html;
            document.getElementById('totalCount').textContent = steps.length;
            updateProgress();
        }}
        
        function onStepInput(el) {{
            const stepNum = parseInt(el.dataset.step);
            const text = el.value.trim();
            
            // Save
            if (!userDescriptions[currentRouteType]) {{
                userDescriptions[currentRouteType] = {{}};
            }}
            userDescriptions[currentRouteType][stepNum] = el.value;
            
            // Visual feedback
            const card = document.getElementById('card-' + stepNum);
            if (text) {{
                el.classList.add('has-text');
                card.classList.add('filled');
            }} else {{
                el.classList.remove('has-text');
                card.classList.remove('filled');
            }}
            
            updateProgress();
        }}
        
        function saveCurrentInputs() {{
            const textareas = document.querySelectorAll('.step-textarea');
            if (!userDescriptions[currentRouteType]) {{
                userDescriptions[currentRouteType] = {{}};
            }}
            textareas.forEach(ta => {{
                const stepNum = parseInt(ta.dataset.step);
                userDescriptions[currentRouteType][stepNum] = ta.value;
            }});
        }}
        
        function updateProgress() {{
            const steps = routeSteps[currentRouteType] || [];
            const saved = userDescriptions[currentRouteType] || {{}};
            
            let filled = 0;
            steps.forEach(step => {{
                const text = (saved[step.step_number] || '').trim();
                if (text) filled++;
            }});
            
            const total = steps.length;
            document.getElementById('filledCount').textContent = filled;
            document.getElementById('totalCount').textContent = total;
            document.getElementById('progressFill').style.width = 
                total > 0 ? (filled / total * 100) + '%' : '0%';
            
            const btn = document.getElementById('submitBtn');
            btn.disabled = filled < total;
        }}
        
        // ─── SUBMIT ───
        function submitDescriptions() {{
            // Kaydet
            saveCurrentInputs();
            
            // Tüm rota tiplerini kontrol et
            const result = {{
                metadata: routeMetadata,
                timestamp: new Date().toISOString(),
                routes: {{}}
            }};
            
            // Aktif rota tipinin verilerini topla
            const steps = routeSteps[currentRouteType] || [];
            const saved = userDescriptions[currentRouteType] || {{}};
            
            // Tamamlanma kontrolü
            let allFilled = true;
            steps.forEach(step => {{
                if (!(saved[step.step_number] || '').trim()) {{
                    allFilled = false;
                }}
            }});
            
            if (!allFilled) {{
                showToast('Please fill in all step descriptions.', 'error');
                return;
            }}
            
            // Sonuç objesi oluştur
            result.routes[currentRouteType] = steps.map(step => ({{
                step_number: step.step_number,
                action: step.action,
                distance_meters: step.distance_meters,
                landmark: step.landmark || null,
                system_description: step.description,
                human_description: (saved[step.step_number] || '').trim()
            }}));
            
            // JSON olarak indir
            const blob = new Blob([JSON.stringify(result, null, 2)], {{ type: 'application/json' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            
            const routeLabel = currentRouteType === 'shortest' ? 'en_kisa' : 'en_az_donus';
            const timestamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, '_');
            a.download = `tarif_${{routeLabel}}_${{timestamp}}.json`;
            a.click();
            URL.revokeObjectURL(url);
            
            showToast('Descriptions saved successfully.', 'success');
        }}
        
        function showToast(message, type) {{
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.className = 'toast ' + type + ' show';
            setTimeout(() => {{ toast.className = 'toast'; }}, 3000);
        }}
    </script>
</body>
</html>'''
    
    return html
