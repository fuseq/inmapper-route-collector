"""
Batch Route Generator - Tüm birimler arasında metrik rota tarifleri oluşturur
"""
import json
import os
from typing import List, Dict, Tuple, Optional
from helpers.path_analysis import distance, compute_connection_problem_scores, build_smooth_path_points, NavigationGraphCleaner
from helpers.dijkstra import dijkstra_connections
from helpers.route_directions import MetricRouteGenerator
from helpers.alternative_routes import generate_alternative_routes
import math


def find_nearest_connection_to_room(room_center: Tuple[float, float], 
                                    graph, 
                                    max_distance: float = 100,
                                    room_id: Optional[str] = None) -> Optional[str]:
    """
    Bir odanın merkez noktasına en yakın connection ID'sini bulur
    
    Öncelik sırası:
    1. Odanın kendi kapısı (door) - connection ID'si room_id ile başlıyorsa
    2. En yakın herhangi bir connection
    
    Args:
        room_center: Odanın merkez koordinatları (x, y)
        graph: Graph nesnesi (connections içeren)
        max_distance: Maksimum kabul edilebilir mesafe
        room_id: Odanın ID'si (kapı eşleştirmesi için, örn: "ID-2402")
    
    Returns:
        En yakın connection'ın ID'si veya None
    """
    nearest_conn_id = None
    min_distance = float('inf')
    
    # 1. ÖNCE: Odanın kendi kapısını (door) ara
    # Kapı ID formatı: room_id + "_" ile başlar (örn: ID-2402_1_, ID-2402_2_)
    if room_id:
        room_door_prefix = room_id + "_"
        room_doors = []
        
        for conn in graph.connections:
            conn_id = conn['id']
            # Connection ID'si oda ID'si ile başlıyorsa bu odanın kapısıdır
            if conn_id.startswith(room_door_prefix):
                conn_mid_x = (conn['x1'] + conn['x2']) / 2
                conn_mid_y = (conn['y1'] + conn['y2']) / 2
                conn_midpoint = (conn_mid_x, conn_mid_y)
                dist = distance(room_center, conn_midpoint)
                room_doors.append((conn_id, dist))
        
        # Odanın kapıları bulunduysa, en yakın olanı döndür
        if room_doors:
            room_doors.sort(key=lambda x: x[1])  # Mesafeye göre sırala
            return room_doors[0][0]
    
    # 2. FALLBACK: Odanın kapısı bulunamadıysa, en yakın connection'ı bul
    for conn in graph.connections:
        # Connection'ın orta noktasını hesapla
        conn_mid_x = (conn['x1'] + conn['x2']) / 2
        conn_mid_y = (conn['y1'] + conn['y2']) / 2
        conn_midpoint = (conn_mid_x, conn_mid_y)
        
        # Mesafeyi hesapla
        dist = distance(room_center, conn_midpoint)
        
        if dist < min_distance and dist <= max_distance:
            min_distance = dist
            nearest_conn_id = conn['id']
    
    # Eğer max_distance içinde bulunamadıysa, en yakın olanı yine de döndür
    if nearest_conn_id is None and graph.connections:
        for conn in graph.connections:
            conn_mid_x = (conn['x1'] + conn['x2']) / 2
            conn_mid_y = (conn['y1'] + conn['y2']) / 2
            conn_midpoint = (conn_mid_x, conn_mid_y)
            dist = distance(room_center, conn_midpoint)
            
            if dist < min_distance:
                min_distance = dist
                nearest_conn_id = conn['id']
    
    return nearest_conn_id


def get_connection_points(graph, conn_id: str) -> Optional[List[Tuple[float, float]]]:
    """
    Connection ID'sine göre başlangıç ve bitiş noktalarını döndürür
    """
    conn = next((c for c in graph.connections if c['id'] == conn_id), None)
    if conn:
        return [(conn['x1'], conn['y1']), (conn['x2'], conn['y2'])]
    return None


def is_significant_turn(angle: float) -> bool:
    """
    Açının anlamlı bir dönüş olup olmadığını kontrol eder
    """
    if angle > 180:
        angle = 360 - angle
    if angle > 175:
        return False
    if angle < 30:
        return False
    return True


def format_angle_direction(prev_point, turn_point, next_point) -> str:
    """
    Dönüş yönünü hesaplar (sola/sağa)
    """
    v1 = (turn_point[0] - prev_point[0], turn_point[1] - prev_point[1])
    v2 = (next_point[0] - turn_point[0], next_point[1] - turn_point[1])
    cross_product = v1[0] * v2[1] - v1[1] * v2[0]
    direction = "sola" if cross_product < 0 else "sağa"
    return direction


def point_to_line_segment_distance(point, p1, p2):
    """Nokta ile çizgi segmenti arasındaki mesafe"""
    x, y = point
    x1, y1 = p1
    x2, y2 = p2
    
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        return math.sqrt((x - x1)**2 + (y - y1)**2)
    
    t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx*dx + dy*dy)))
    
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    
    return math.sqrt((x - proj_x)**2 + (y - proj_y)**2)


def point_in_polygon(point, polygon):
    """Ray casting algoritması ile noktanın polygon içinde olup olmadığını kontrol et"""
    x, y = point
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    
    return inside


def distance_to_polygon(point, polygon):
    """Noktanın polygon'a olan minimum mesafesi (içerideyse 0)"""
    if not polygon or len(polygon) < 3:
        return float('inf')
    
    if point_in_polygon(point, polygon):
        return 0.0
    
    # Kenarlar üzerindeki en yakın noktayı bul
    min_dist = float('inf')
    n = len(polygon)
    
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        
        dist = point_to_line_segment_distance(point, p1, p2)
        if dist < min_dist:
            min_dist = dist
    
    return min_dist


def find_nearest_largest_room(turn_point, room_areas, max_distance=500, pixel_to_meter=0.1, graph=None,
                              prev_point=None, next_point=None, end_point=None):
    """
    Dönüş noktası için EN YAKIN odayı bulur (polygon mesafesi kullanarak)
    
    Basit mantık:
    - Odanın polygon sınırına olan mesafe hesaplanır
    - Nokta odanın içindeyse mesafe = 0
    - Daima en yakın oda seçilir
    - Anchor mesafesi, dönüş noktasından hedefe olan mesafeden fazla olamaz
    - Otopark alanları için: Doors grubundaki carpark-* door'ları kullanılır
    
    Args:
        turn_point: Dönüş noktası koordinatları
        room_areas: Oda bilgileri dictionary
        max_distance: Maksimum arama mesafesi (piksel)
        pixel_to_meter: Piksel-metre dönüşüm oranı
        graph: Graph nesnesi (otopark door'ları için)
        end_point: Hedef nokta (mesafe limiti için)
    
    Returns:
        (room_type, room_id, area, distance) veya None
    """
    # Hedefe olan mesafeyi hesapla (varsa)
    distance_to_end = float('inf')
    if end_point:
        dx = end_point[0] - turn_point[0]
        dy = end_point[1] - turn_point[1]
        distance_to_end = math.sqrt(dx*dx + dy*dy)
    
    # Kullanılabilir maksimum mesafe (piksel cinsinden)
    effective_max_distance = min(max_distance, distance_to_end)
    
    best_room = None
    best_distance = float('inf')
    
    # Anchor olarak kullanılabilecek oda tipleri
    # Shop, Food, Medical, Commercial, Social = navigasyonda referans olabilecek yerler
    # Building, Walking, Water, Green, Other = navigasyonda referans olmaz
    allowed_types = {'Shop', 'Food', 'Medical', 'Commercial', 'Social'}
    
    # En yakın odaları topla
    candidates = []
    
    for room_type, rooms in room_areas.items():
        # Sadece izin verilen tipleri kullan
        if room_type not in allowed_types:
            continue
            
        for room in rooms:
            center = room.get('center')
            if not center:
                continue
            
            # Polygon varsa polygon mesafesi, yoksa center mesafesi
            polygon = room.get('coordinates')
            if polygon and len(polygon) >= 3:
                dist = distance_to_polygon(turn_point, polygon)
            else:
                # Polygon yoksa center'a mesafe
                dx = center[0] - turn_point[0]
                dy = center[1] - turn_point[1]
                dist = math.sqrt(dx*dx + dy*dy)
            
            # Mesafe kontrolü (center bazlı ön filtre - polygon içindeyse dist=0)
            if dist > effective_max_distance:
                continue
            
            candidates.append({
                'type': room_type,
                'id': room.get('id', ''),
                'area': room.get('area', 0),
                'distance': dist,
                'center': center
            })
            
            # En yakın olanı seç
            if dist < best_distance:
                best_distance = dist
                best_room = {
                    'type': room_type,
                    'id': room.get('id', ''),
                    'area': room.get('area', 0),
                    'distance': dist
                }
    
    # OTOPARK ANCHOR'LARI: Graph'taki carpark-* door'larını kontrol et
    if graph and hasattr(graph, 'connections'):
        for conn in graph.connections:
            conn_id = conn.get('id', '')
            conn_type = conn.get('type', '')
            
            # Sadece carpark- ile başlayan door'ları kontrol et
            if conn_type == 'door' and conn_id.startswith('carpark-'):
                # Door'un merkez noktasını hesapla
                door_center_x = (conn['x1'] + conn['x2']) / 2
                door_center_y = (conn['y1'] + conn['y2']) / 2
                door_center = (door_center_x, door_center_y)
                
                # Mesafe hesapla
                dx = door_center_x - turn_point[0]
                dy = door_center_y - turn_point[1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Mesafe kontrolü
                if dist > effective_max_distance:
                    continue
                
                # Door uzunluğunu alan olarak kullan (yaklaşık)
                door_length = math.sqrt(
                    (conn['x2'] - conn['x1'])**2 + 
                    (conn['y2'] - conn['y1'])**2
                )
                
                # Orijinal door ID'sini kullan (SVG'de highlight için gerekli)
                # carpark-227_1_ formatında sakla
                candidates.append({
                    'type': 'Parking',
                    'id': conn_id,  # Orijinal ID: carpark-227_1_
                    'area': door_length * 100,  # Yaklaşık alan
                    'distance': dist,
                    'center': door_center
                })
                
                # En yakın olanı seç
                if dist < best_distance:
                    best_distance = dist
                    best_room = {
                        'type': 'Parking',
                        'id': conn_id,  # Orijinal ID: carpark-227_1_
                        'area': door_length * 100,
                        'distance': dist
                    }
    
    # PORTAL ANCHOR'LARI: Graph'taki portal'ları kontrol et (Asansör, Merdiven)
    if graph and hasattr(graph, 'connections'):
        for conn in graph.connections:
            conn_id = conn.get('id', '')
            conn_type = conn.get('type', '')
            
            # Sadece portal tipindeki connection'ları kontrol et
            if conn_type == 'portal':
                # Portal'un merkez noktasını hesapla
                portal_center_x = (conn['x1'] + conn['x2']) / 2
                portal_center_y = (conn['y1'] + conn['y2']) / 2
                portal_center = (portal_center_x, portal_center_y)
                
                # Mesafe hesapla
                dx = portal_center_x - turn_point[0]
                dy = portal_center_y - turn_point[1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Mesafe kontrolü
                if dist > effective_max_distance:
                    continue
                
                # Portal uzunluğunu alan olarak kullan (yaklaşık)
                portal_length = math.sqrt(
                    (conn['x2'] - conn['x1'])**2 + 
                    (conn['y2'] - conn['y1'])**2
                )
                
                # Portal tipini belirle (Asansör/Merdiven)
                if 'Elev' in conn_id:
                    portal_type = 'Elevator'
                elif 'Stair' in conn_id:
                    portal_type = 'Stairs'
                else:
                    portal_type = 'Portal'
                
                candidates.append({
                    'type': portal_type,
                    'id': conn_id,  # Orijinal ID
                    'area': portal_length * 100,  # Yaklaşık alan
                    'distance': dist,
                    'center': portal_center
                })
                
                # En yakın olanı seç
                if dist < best_distance:
                    best_distance = dist
                    best_room = {
                        'type': portal_type,
                        'id': conn_id,  # Orijinal ID
                        'area': portal_length * 100,
                        'distance': dist
                    }
    
    if best_room:
        return (best_room['type'], best_room['id'], best_room['area'], best_room['distance'])
    
    return None


# Gerçek dönüş ile koridor kırılımı (bend) ayrımı için eşik
BEND_THRESHOLD = 55  # Derece: bunun altı = bend (landmark referansı), üstü = gerçek dönüş


def _find_room_center(room_id: str, floor_areas: Dict) -> Optional[Tuple[float, float]]:
    """floor_areas içinden bir odanın merkez koordinatını bul"""
    for rtype, rooms in floor_areas.items():
        for room in rooms:
            if room.get('id') == room_id:
                return room.get('center')
    return None


def _compute_anchor_side(current_point, prev_point, next_point, 
                         anchor_center) -> Optional[str]:
    """
    Anchor'ın yürüme yönüne göre hangi tarafta olduğunu hesapla.
    
    SVG koordinat sistemi (Y aşağı):
    - cross > 0 → anchor sağda
    - cross < 0 → anchor solda
    """
    # Yürüme yönü: önceki noktadan sonraki noktaya
    walk_dx = next_point[0] - prev_point[0]
    walk_dy = next_point[1] - prev_point[1]
    walk_mag = math.hypot(walk_dx, walk_dy)
    
    if walk_mag < 1.0:
        return None
    
    walk_dx /= walk_mag
    walk_dy /= walk_mag
    
    # Anchor yönü: mevcut noktadan anchor merkezine
    anc_dx = anchor_center[0] - current_point[0]
    anc_dy = anchor_center[1] - current_point[1]
    
    # Çapraz çarpım
    cross = walk_dx * anc_dy - walk_dy * anc_dx
    
    if abs(cross) < 0.1:
        return None
    
    return 'sol' if cross < 0 else 'sag'


def _consolidate_zigzags(turns, path_points, floor_areas):
    """
    Ardışık zıt yönlü yakın dönüşleri (zigzag) tespit edip tek bir 'veer' 
    (hafif yönelme) talimatına dönüştürür.
    
    Zigzag = Kısa mesafede sola→sağa veya sağa→sola dönüş çifti.
    Net yön değişimi küçükse → 2 dönüş yerine 1 "hafif sola/sağa yönelin" talimatı.
    
    Örnek: ID012'de sola 106° + sağa 100° = net ~6° sola
           → "Hafif sola yönelerek devam edin"
    """
    if len(turns) < 2:
        return turns
    
    result = []
    i = 0
    
    while i < len(turns):
        merged = False
        
        if i < len(turns) - 1:
            t1 = turns[i]
            t2 = turns[i + 1]
            
            # 1) Zıt yönler mi? (sola ↔ sağa)
            if t1['direction'] != t2['direction']:
                p1 = t1['point']
                p2 = t2['point']
                
                idx1 = t1['path_index']
                idx2 = t2['path_index']
                
                # 2) Path üzerindeki yürüme mesafesi kısa mı?
                path_dist = 0.0
                for j in range(idx1, idx2):
                    pj = path_points[j]
                    pk = path_points[j + 1]
                    path_dist += math.hypot(pk[0] - pj[0], pk[1] - pj[1])
                
                # 3) Açı büyüklükleri benzer mi?
                #    Gerçek zigzag: 106° + 100° → oran 0.94 ✓
                #    Farklı tip: 89° + 46° → oran 0.51 ✗
                angle1 = t1['angle']
                angle2 = t2['angle']
                max_angle = max(angle1, angle2)
                angle_ratio = min(angle1, angle2) / max_angle if max_angle > 0 else 0
                
                # Çok kısa mesafede (< 15px ≈ 1.5m) açı oranı daha esnek
                # çünkü bu kadar kısa mesafede herhangi bir yön değişimi zigzag'dır
                angle_ratio_threshold = 0.3 if path_dist < 15 else 0.6
                
                # Tüm koşullar sağlanırsa zigzag
                if (angle_ratio >= angle_ratio_threshold and 
                    path_dist < 50 and 
                    idx1 > 0 and idx2 < len(path_points) - 1):
                    
                    at1 = path_points[idx1]
                    at2 = path_points[idx2]
                    
                    # Kararlı giriş yönü: zigzag'dan en az 30px önceki noktayı bul
                    MIN_STABLE_DIST = 30.0
                    stable_before = path_points[idx1 - 1]
                    cumul = 0.0
                    for k in range(idx1, 0, -1):
                        seg = math.hypot(path_points[k][0] - path_points[k-1][0],
                                        path_points[k][1] - path_points[k-1][1])
                        cumul += seg
                        if cumul >= MIN_STABLE_DIST:
                            stable_before = path_points[k - 1]
                            break
                    
                    # Kararlı çıkış yönü: zigzag'dan en az 30px sonraki noktayı bul
                    stable_after = path_points[idx2 + 1]
                    cumul = 0.0
                    for k in range(idx2, len(path_points) - 1):
                        seg = math.hypot(path_points[k+1][0] - path_points[k][0],
                                        path_points[k+1][1] - path_points[k][1])
                        cumul += seg
                        if cumul >= MIN_STABLE_DIST:
                            stable_after = path_points[k + 1]
                            break
                    
                    # Giriş-çıkış bearing farkı (kararlı noktalarla)
                    bearing_in = math.atan2(at1[1] - stable_before[1], at1[0] - stable_before[0])
                    bearing_out = math.atan2(stable_after[1] - at2[1], stable_after[0] - at2[0])
                    
                    diff = bearing_out - bearing_in
                    while diff > math.pi:
                        diff -= 2 * math.pi
                    while diff < -math.pi:
                        diff += 2 * math.pi
                    
                    diff_deg = abs(math.degrees(diff))
                    
                    if diff_deg < 45:
                        # Net yönü belirle: fiziksel kayma yönü (zigzag path'i hangi tarafa kaydırıyor?)
                        # Approach yönü × zigzag jog displacement → sağa mı sola mı kaydı?
                        v_approach = (at1[0] - stable_before[0], at1[1] - stable_before[1])
                        v_jog = (at2[0] - at1[0], at2[1] - at1[1])
                        cross = v_approach[0] * v_jog[1] - v_approach[1] * v_jog[0]
                        
                        if abs(cross) > 0.5:
                            net_direction = 'sağa' if cross > 0 else 'sola'
                        else:
                            net_direction = 'düz'
                        
                        # Anchor: her iki turn'dan birini al
                        anchor = t1.get('anchor') or t2.get('anchor')
                        
                        # Anchor side hesapla (kararlı noktalarla)
                        anchor_side = None
                        if anchor:
                            room_id = anchor[1]
                            anchor_center = _find_room_center(room_id, floor_areas)
                            if anchor_center:
                                mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                                anchor_side = _compute_anchor_side(
                                    mid_point, stable_before, stable_after, anchor_center
                                )
                        
                        print(f"    [Zigzag→Veer] {t1['direction']} {t1['angle']:.1f}° + "
                              f"{t2['direction']} {t2['angle']:.1f}° → net {diff_deg:.1f}° "
                              f"({net_direction}) [path: {path_dist:.0f}px]")
                        
                        veer_turn = {
                            'point': p1,
                            'path_index': idx1,
                            'angle': diff_deg,
                            'direction': net_direction,
                            'anchor': anchor,
                            'turn_type': 'veer',
                            'anchor_side': anchor_side
                        }
                        result.append(veer_turn)
                        i += 2
                        merged = True
        
        if not merged:
            result.append(turns[i])
            i += 1
    
    return result


def _merge_adjacent_micro_turns(turns, path_points, floor_areas):
    """
    İlk zigzag konsolidasyonundan sonra arta kalan mikro-segment sorunlarını temizler.
    
    İki durum ele alınır:
    
    1) VEER + aynı yönlü TURN çok yakınsa (< 50px):
       VEER sadece gürültüdür, absorbe edilir.
       TURN orijinal konumunda kalır ama turn_type='bend' yapılır
       → PASS_BY olarak gösterilir, anchor referansı korunur.
    
    2) Ardışık iki dönüş (turn/bend) arası çok kısa segment (< 20px ≈ 2m):
       Zıt yön → VEER'e dönüştür
       Aynı yön → büyük açılı olanı tut
    """
    if len(turns) < 2:
        return turns
    
    MICRO_DIST = 20.0   # px — mikro segment eşiği (~2m)
    VEER_ABSORB = 50.0  # px — VEER absorbe mesafesi (~5m)
    
    result = []
    i = 0
    
    print(f"    [MergeDebug] Giriş: {len(turns)} turn")
    for _d, _t in enumerate(turns):
        _a = f"{_t.get('anchor', ('?','?'))[1]}" if _t.get('anchor') else '?'
        print(f"      [{_d}] {_t.get('turn_type')} {_t['direction']} {_t['angle']:.1f}° idx={_t['path_index']} anc={_a}")
    
    while i < len(turns):
        if i < len(turns) - 1:
            t1 = turns[i]
            t2 = turns[i + 1]
            
            idx1 = t1['path_index']
            idx2 = t2['path_index']
            
            # İki dönüş arası path mesafesi
            seg_dist = 0.0
            for j in range(idx1, min(idx2, len(path_points) - 1)):
                seg_dist += distance(path_points[j], path_points[j + 1])
            
            print(f"    [MergeDebug] i={i}: {t1.get('turn_type')}({t1['direction']}) + {t2.get('turn_type')}({t2['direction']}) seg={seg_dist:.0f}px")
            
            # ── Durum 1: VEER + aynı yönlü TURN/BEND (çok yakın) ──
            # Örnek: VEER(sola 6°) + TURN(sola 67.8°) @ 30px
            # VEER gürültüdür (zigzag kalıntısı). TURN orijinal yerinde kalır
            # ama bend olarak gösterilir → PASS_BY (anchor referansı korunur).
            if (t1.get('turn_type') == 'veer' and 
                t2.get('turn_type') in ('turn', 'bend') and
                seg_dist < VEER_ABSORB):
                # Aynı yön veya düz VEER → absorbe et
                if (t1['direction'] == t2['direction'] or 
                    t1['direction'] == 'düz' or
                    t1['angle'] < 15):
                    # TURN orijinal konumunda kalır, bend'e dönüşür
                    merged_turn = dict(t2)
                    merged_turn['turn_type'] = 'bend'
                    # anchor_side hesapla (bend için gerekli)
                    if merged_turn.get('anchor') and not merged_turn.get('anchor_side'):
                        room_id = merged_turn['anchor'][1]
                        anchor_center = _find_room_center(room_id, floor_areas)
                        if anchor_center:
                            prev_pt = path_points[max(0, idx2 - 1)]
                            next_pt = path_points[min(len(path_points) - 1, idx2 + 1)]
                            merged_turn['anchor_side'] = _compute_anchor_side(
                                merged_turn['point'], prev_pt, next_pt, anchor_center
                            )
                    
                    print(f"    [Veer+Turn→Bend] VEER({t1['direction']} {t1['angle']:.1f}°) "
                          f"absorbe → BEND({t2['direction']} {t2['angle']:.1f}°) "
                          f"@ {t2.get('anchor', ('?','?'))[1]} [seg: {seg_dist:.0f}px]")
                    result.append(merged_turn)
                    i += 2
                    continue
            
            # ── Durum 1b: TURN/BEND + aynı yönlü VEER (çok yakın) ──
            if (t2.get('turn_type') == 'veer' and 
                t1.get('turn_type') in ('turn', 'bend') and
                seg_dist < VEER_ABSORB):
                if (t1['direction'] == t2['direction'] or 
                    t2['direction'] == 'düz' or
                    t2['angle'] < 15):
                    # TURN'ü tut, VEER'i absorbe et
                    result.append(t1)
                    i += 2
                    continue
            
            # ── Durum 2: Mikro segment (< 20px) — herhangi iki dönüş ──
            # Ama büyük açılı gerçek dönüşleri (≥ 70°) koruyoruz
            if (seg_dist < MICRO_DIST and 
                t1['angle'] < 70 and t2['angle'] < 70):
                dir1 = t1['direction']
                dir2 = t2['direction']
                
                if dir1 != dir2 and dir1 != 'düz' and dir2 != 'düz':
                    # Zıt yönler → VEER'e dönüştür
                    # Net yönü ve açıyı hesapla
                    net_angle = abs(t1['angle'] - t2['angle'])
                    net_dir = dir1 if t1['angle'] > t2['angle'] else dir2
                    
                    anchor = t1.get('anchor') or t2.get('anchor')
                    anchor_side = None
                    if anchor:
                        room_id = anchor[1]
                        anchor_center = _find_room_center(room_id, floor_areas)
                        if anchor_center:
                            mid = ((t1['point'][0]+t2['point'][0])/2,
                                   (t1['point'][1]+t2['point'][1])/2)
                            # Yürüme yönü: idx1'den sonraki kararlı nokta
                            before_pt = path_points[max(0, idx1-1)]
                            after_pt = path_points[min(len(path_points)-1, idx2+1)]
                            anchor_side = _compute_anchor_side(
                                mid, before_pt, after_pt, anchor_center
                            )
                    
                    print(f"    [Micro→Veer] {dir1} {t1['angle']:.1f}° + "
                          f"{dir2} {t2['angle']:.1f}° → net {net_angle:.1f}° "
                          f"({net_dir}) [seg: {seg_dist:.0f}px]")
                    
                    veer = {
                        'point': t1['point'],
                        'path_index': idx1,
                        'angle': net_angle,
                        'direction': net_dir,
                        'anchor': anchor,
                        'turn_type': 'veer',
                        'anchor_side': anchor_side
                    }
                    result.append(veer)
                    i += 2
                    continue
                else:
                    # Aynı yön → açıları birleştir
                    combined_angle = t1['angle'] + t2['angle']
                    keep = t1 if t1['angle'] >= t2['angle'] else t2
                    merged = dict(keep)
                    merged['angle'] = combined_angle
                    
                    # Birleşik açı BEND_THRESHOLD üstündeyse → gerçek dönüş
                    if combined_angle >= BEND_THRESHOLD:
                        merged['turn_type'] = 'turn'
                        print(f"    [Micro→Turn] {dir1} {t1['angle']:.1f}° + "
                              f"{t2['angle']:.1f}° = {combined_angle:.1f}° "
                              f"(>= {BEND_THRESHOLD}° → TURN) [seg: {seg_dist:.0f}px]")
                    else:
                        print(f"    [Micro→Keep] {dir1} {t1['angle']:.1f}° + "
                              f"{t2['angle']:.1f}° = {combined_angle:.1f}° "
                              f"[seg: {seg_dist:.0f}px]")
                    
                    result.append(merged)
                    i += 2
                    continue
        
        result.append(turns[i])
        i += 1
    
    return result


def _find_door_junction_points(path_conn_ids, graph):
    """
    Path'in başında ve sonunda kapı (door) connection'larının ilk path ile
    buluştuğu junction noktalarını bulur.
    
    Sadece kapı çıkış geometrisini filtreler (kapı ile ilk path arasındaki
    kısa segment junction'ı). Koridordaki gerçek dönüşlere dokunmaz.
    
    Returns:
        (start_junction, end_junction): Her biri (x, y) tuple veya None
    """
    if not path_conn_ids or not graph:
        return None, None
    
    conn_lookup = {c['id']: c for c in graph.connections}
    
    # ── Başlangıçtaki kapı junction'ı ──
    start_junction = None
    for i, conn_id in enumerate(path_conn_ids):
        conn = conn_lookup.get(conn_id)
        if not conn:
            continue
        if conn['type'] == 'door':
            continue
        # İlk koridor connection bulundu — önceki kapıyla buluşma noktası
        if i > 0:
            prev_conn = conn_lookup.get(path_conn_ids[i - 1])
            if prev_conn:
                # Koridor connection'ın kapıya yakın ucu = junction
                p1 = (conn['x1'], conn['y1'])
                p2 = (conn['x2'], conn['y2'])
                prev_ends = [(prev_conn['x1'], prev_conn['y1']),
                             (prev_conn['x2'], prev_conn['y2'])]
                d_p1 = min(distance(p1, pe) for pe in prev_ends)
                d_p2 = min(distance(p2, pe) for pe in prev_ends)
                start_junction = p1 if d_p1 < d_p2 else p2
        break
    
    # ── Bitişteki kapı junction'ı ──
    end_junction = None
    for j in range(len(path_conn_ids) - 1, -1, -1):
        conn = conn_lookup.get(path_conn_ids[j])
        if not conn:
            continue
        if conn['type'] == 'door':
            continue
        # Son koridor connection bulundu — sonraki kapıyla buluşma noktası
        if j < len(path_conn_ids) - 1:
            next_conn = conn_lookup.get(path_conn_ids[j + 1])
            if next_conn:
                p1 = (conn['x1'], conn['y1'])
                p2 = (conn['x2'], conn['y2'])
                next_ends = [(next_conn['x1'], next_conn['y1']),
                             (next_conn['x2'], next_conn['y2'])]
                d_p1 = min(distance(p1, ne) for ne in next_ends)
                d_p2 = min(distance(p2, ne) for ne in next_ends)
                end_junction = p1 if d_p1 < d_p2 else p2
        break
    
    return start_junction, end_junction


def _detect_turns_from_path(path_points, path_conn_ids, graph, floor_areas):
    """
    Path noktalarından dönüş noktalarını tespit eder.
    
    İki tip tespit:
    - 'turn' (>= BEND_THRESHOLD°): Gerçek dönüş → "Sağa/Sola dönün"
    - 'bend' (30° - BEND_THRESHOLD°): Koridor kırılımı → "X sağınızda/solunuzda kalacak şekilde ilerleyin"
      (Sadece yakında anchor varsa dahil edilir, yoksa atlanır)
    
    Kapı segmentlerindeki dönüşler filtrelenir:
    - Başlangıçta: kapı çıkışı → koridor geçişindeki dönüş (START ile birleştirilir)
    - Bitişte: koridor → kapı girişindeki dönüş (ARRIVE ile birleştirilir)
    
    Sonrasında zigzag konsolidasyonu uygulanır:
    - Ardışık zıt yönlü kısa mesafeli dönüş çiftleri (net yön değişimi < 45°) kaldırılır.
    """
    # Kapı junction noktalarını bul (başlangıç ve bitiş)
    start_door_jn, end_door_jn = _find_door_junction_points(path_conn_ids, graph)
    DOOR_JUNCTION_TOLERANCE = 15.0  # px — junction noktasına bu kadar yakın dönüşler atlanır
    
    all_turns = []
    for i in range(1, len(path_points) - 1):
        prev_point = path_points[i - 1]
        current_point = path_points[i]
        next_point = path_points[i + 1]
        
        v1 = (current_point[0] - prev_point[0], current_point[1] - prev_point[1])
        v2 = (next_point[0] - current_point[0], next_point[1] - current_point[1])
        
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        
        if mag1 == 0 or mag2 == 0:
            continue
        
        cos_theta = dot / (mag1 * mag2)
        cos_theta = max(min(cos_theta, 1), -1)
        angle = math.acos(cos_theta) * (180 / math.pi)
        
        if is_significant_turn(angle):
            # ── Kapı junction filtresi ──
            # Başlangıçta kapı çıkışı veya bitişte kapı girişindeki dönüşleri atla.
            # Bu dönüşler navigasyon kararı değil, sadece oda giriş/çıkış geometrisi.
            if start_door_jn and distance(current_point, start_door_jn) < DOOR_JUNCTION_TOLERANCE:
                continue
            if end_door_jn and distance(current_point, end_door_jn) < DOOR_JUNCTION_TOLERANCE:
                continue
            
            direction = format_angle_direction(prev_point, current_point, next_point)
            
            # Gerçek dönüş mü, koridor kırılımı mı?
            turn_type = 'turn' if angle >= BEND_THRESHOLD else 'bend'
            
            # En yakın odayı bul
            end_point = path_points[-1]
            nearest_room = find_nearest_largest_room(
                current_point, 
                floor_areas,
                max_distance=400,
                end_point=end_point,
                graph=graph
            )
            
            # Bend ise ve anchor yoksa → atla (kullanıcıya bilgi vermez)
            if turn_type == 'bend' and not nearest_room:
                continue
            
            # Bend'ler için anchor'ın hangi tarafta olduğunu hesapla
            anchor_side = None
            if turn_type == 'bend' and nearest_room:
                room_type, room_id, area, dist = nearest_room
                anchor_center = _find_room_center(room_id, floor_areas)
                if anchor_center:
                    anchor_side = _compute_anchor_side(
                        current_point, prev_point, next_point, anchor_center
                    )
            
            turn_info = {
                'point': current_point,
                'path_index': i,
                'angle': angle,
                'direction': direction,
                'anchor': nearest_room,
                'turn_type': turn_type,
                'anchor_side': anchor_side
            }
            all_turns.append(turn_info)
    
    # Zigzag konsolidasyonu: ardışık zıt yönlü kısa dönüşleri hafif yönelmeye dönüştür
    all_turns = _consolidate_zigzags(all_turns, path_points, floor_areas)
    
    # İkinci geçiş: VEER/turn birleştirme — ilk konsolidasyondan arta kalan
    # orphan dönüşleri (VEER + aynı yönlü turn gibi) temizle
    all_turns = _merge_adjacent_micro_turns(all_turns, path_points, floor_areas)
    
    return all_turns


def generate_alternative_routes_for_room_pair(start_room: Dict,
                                               end_room: Dict,
                                               graph,
                                               floor_areas: Dict,
                                               pixel_to_meter_ratio: float = 0.1) -> Optional[Dict]:
    """
    İki oda arasında alternatif rotalar oluşturur
    
    Args:
        start_room: Başlangıç odası bilgisi
        end_room: Hedef odası bilgisi
        graph: Graph nesnesi
        floor_areas: Kat alanları bilgisi
        pixel_to_meter_ratio: Piksel-metre oranı
    
    Returns:
        Alternatif rotalar dictionary
    """
    # Odalara en yakın connection'ları bul (önce odanın kendi kapısını ara)
    start_conn_id = find_nearest_connection_to_room(start_room['center'], graph, room_id=start_room['id'])
    end_conn_id = find_nearest_connection_to_room(end_room['center'], graph, room_id=end_room['id'])
    
    if not start_conn_id or not end_conn_id:
        return None
    
    if start_conn_id == end_conn_id:
        return None
    
    # Alternatif rotaları hesapla
    alternative_routes = generate_alternative_routes(graph, start_conn_id, end_conn_id)
    
    if not alternative_routes:
        return None
    
    # Her alternatif için detaylı bilgi oluştur
    result = {
        'from': {
            'type': start_room['type'],
            'id': start_room['id'],
            'center': start_room['center']
        },
        'to': {
            'type': end_room['type'],
            'id': end_room['id'],
            'center': end_room['center']
        },
        'routes': {}
    }
    
    # Navigation graph cleaner oluştur (veya cache'den al)
    if not hasattr(graph, '_nav_cleaner'):
        graph._nav_cleaner = NavigationGraphCleaner(graph)
    
    for route_type, route_data in alternative_routes.items():
        path = route_data['path']
        
        # Chain-based temizlenmiş path_points oluştur
        path_points = graph._nav_cleaner.clean_path(path)
        
        # Dönüş noktalarını bul
        all_turns = _detect_turns_from_path(path_points, path, graph, floor_areas)
        
        # Metrik tarif oluştur
        metric_generator = MetricRouteGenerator(pixel_to_meter_ratio=pixel_to_meter_ratio)
        start_location = f"{start_room['type']} - {start_room['id']}"
        end_location = f"{end_room['type']} - {end_room['id']}"
        
        metric_generator.generate_directions(
            path_points=path_points,
            turns=all_turns,
            start_location=start_location,
            end_location=end_location,
            start_room_center=start_room['center'],
            path_conn_ids=path,
            graph=graph
        )
        
        summary = metric_generator.get_summary()
        
        real_turns_count = sum(1 for t in all_turns if t.get('turn_type', 'turn') == 'turn')
        
        result['routes'][route_type] = {
            'name': route_data['name'],
            'description': route_data['description'],
            'summary': summary,
            'steps': metric_generator.export_to_json(),
            'path_connections': path,
            'path_points': path_points,
            'turns': all_turns,
            'turns_count': real_turns_count,
            'metrics': route_data['metrics']
        }
    
    return result


def generate_route_for_room_pair(start_room: Dict, 
                                 end_room: Dict,
                                 graph,
                                 floor_areas: Dict,
                                 pixel_to_meter_ratio: float = 0.1) -> Optional[Dict]:
    """
    İki oda arasında rota oluşturur
    
    Args:
        start_room: Başlangıç odası bilgisi {'id': ..., 'center': ..., 'type': ...}
        end_room: Hedef odası bilgisi
        graph: Graph nesnesi
        floor_areas: Kat alanları bilgisi
        pixel_to_meter_ratio: Piksel-metre oranı
    
    Returns:
        Rota bilgilerini içeren dictionary veya None
    """
    # Odalara en yakın connection'ları bul (önce odanın kendi kapısını ara)
    start_conn_id = find_nearest_connection_to_room(start_room['center'], graph, room_id=start_room['id'])
    end_conn_id = find_nearest_connection_to_room(end_room['center'], graph, room_id=end_room['id'])
    
    if not start_conn_id or not end_conn_id:
        return None
    
    # Aynı oda ise atla
    if start_conn_id == end_conn_id:
        return None
    
    # Dijkstra ile rota hesapla
    try:
        path = dijkstra_connections(graph, start_conn_id, end_conn_id)
    except Exception as e:
        print(f"Rota hesaplanamadı: {start_room['id']} -> {end_room['id']}: {str(e)}")
        return None
    
    if not path or len(path) < 2:
        return None
    
    # Navigation graph cleaner oluştur (veya cache'den al)
    if not hasattr(graph, '_nav_cleaner'):
        graph._nav_cleaner = NavigationGraphCleaner(graph)
    
    # Chain-based temizlenmiş path_points oluştur
    path_points = graph._nav_cleaner.clean_path(path)
    
    if len(path_points) < 2:
        return None
    
    # Dönüş noktalarını analiz et
    all_turns = _detect_turns_from_path(path_points, path, graph, floor_areas)
    
    # Metrik rota tarifini oluştur
    metric_generator = MetricRouteGenerator(pixel_to_meter_ratio=pixel_to_meter_ratio)
    
    start_location = f"{start_room['type']} - {start_room['id']}"
    end_location = f"{end_room['type']} - {end_room['id']}"
    
    metric_steps = metric_generator.generate_directions(
        path_points=path_points,
        turns=all_turns,
        start_location=start_location,
        end_location=end_location,
        start_room_center=start_room['center'],
        path_conn_ids=path,
        graph=graph
    )
    
    # Özet bilgileri al
    summary = metric_generator.get_summary()
    
    real_turns_count = sum(1 for t in all_turns if t.get('turn_type', 'turn') == 'turn')
    
    return {
        'from': {
            'type': start_room['type'],
            'id': start_room['id'],
            'center': start_room['center']
        },
        'to': {
            'type': end_room['type'],
            'id': end_room['id'],
            'center': end_room['center']
        },
        'summary': summary,
        'steps': metric_generator.export_to_json(),
        'path_connections': path,
        'path_points': path_points,
        'turns': all_turns,
        'turns_count': real_turns_count
    }


def generate_all_routes_for_floor(graph, 
                                  floor_areas: Dict,
                                  floor_name: str,
                                  output_dir: str = "routes",
                                  pixel_to_meter_ratio: float = 0.1,
                                  max_routes: Optional[int] = None) -> Dict:
    """
    Bir kat için tüm birimler arası rotaları oluşturur
    
    Args:
        graph: Graph nesnesi
        floor_areas: Kat alanları dictionary
        floor_name: Kat adı (örn: "Kat_0", "-1")
        output_dir: Çıktı dizini
        pixel_to_meter_ratio: Piksel-metre oranı
        max_routes: Maksimum rota sayısı (test için, None = tümü)
    
    Returns:
        Oluşturulan rota istatistikleri
    """
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # Tüm odaları tek bir listeye topla
    all_rooms = []
    for room_type, rooms in floor_areas.items():
        for room in rooms:
            all_rooms.append({
                'id': room['id'],
                'type': room_type,
                'center': room['center'],
                'area': room['area']
            })
    
    print(f"\n{'='*70}")
    print(f"KAT: {floor_name}")
    print(f"Toplam Birim Sayısı: {len(all_rooms)}")
    print(f"Toplam Olası Rota Sayısı: {len(all_rooms) * (len(all_rooms) - 1)}")
    print(f"{'='*70}\n")
    
    # Tüm rota kombinasyonları
    routes = {}
    successful_routes = 0
    failed_routes = 0
    route_count = 0
    
    for i, start_room in enumerate(all_rooms):
        for j, end_room in enumerate(all_rooms):
            if i == j:  # Aynı oda
                continue
            
            # Max route limiti kontrolü
            if max_routes and route_count >= max_routes:
                break
            
            route_key = f"{start_room['type']}_{start_room['id']}_to_{end_room['type']}_{end_room['id']}"
            
            print(f"Rota hesaplanıyor ({route_count + 1}): {start_room['type']}-{start_room['id']} -> {end_room['type']}-{end_room['id']}")
            
            route_data = generate_route_for_room_pair(
                start_room=start_room,
                end_room=end_room,
                graph=graph,
                floor_areas=floor_areas,
                pixel_to_meter_ratio=pixel_to_meter_ratio
            )
            
            if route_data:
                routes[route_key] = route_data
                successful_routes += 1
                print(f"  ✓ Başarılı - Mesafe: {route_data['summary']['total_distance_meters']:.1f}m, Dönüş: {route_data['turns_count']}")
            else:
                failed_routes += 1
                print(f"  ✗ Rota bulunamadı")
            
            route_count += 1
        
        if max_routes and route_count >= max_routes:
            break
    
    # Sonuçları JSON'a kaydet
    output_file = os.path.join(output_dir, f"routes_{floor_name}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'floor': floor_name,
            'total_rooms': len(all_rooms),
            'successful_routes': successful_routes,
            'failed_routes': failed_routes,
            'routes': routes
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"ÖZET - {floor_name}")
    print(f"{'='*70}")
    print(f"Başarılı Rotalar: {successful_routes}")
    print(f"Başarısız Rotalar: {failed_routes}")
    print(f"Dosya: {output_file}")
    print(f"{'='*70}\n")
    
    return {
        'floor': floor_name,
        'total_rooms': len(all_rooms),
        'successful_routes': successful_routes,
        'failed_routes': failed_routes,
        'output_file': output_file
    }


