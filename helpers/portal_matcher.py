"""
Portal Matcher - Katlar arası geçiş için portal eşleştirme ve yol bulma
"""
import re
from typing import List, Dict, Tuple, Optional


def parse_portal_name(portal_id: str) -> Optional[Dict]:
    """
    Portal isminden bilgi çıkarır
    
    Örnek: 'Elev.8.1' -> {'type': 'Elev', 'number': '8', 'target_floor': '1'}
    Örnek: 'Stairs.3.-1' -> {'type': 'Stairs', 'number': '3', 'target_floor': '-1'}
    
    Args:
        portal_id: Portal ID'si
    
    Returns:
        Portal bilgilerini içeren dict veya None
    """
    # Pattern: Type.Number.TargetFloor
    # TargetFloor negatif olabilir (-1, -2, -3)
    pattern = r'^(Elev|Stairs)\.(\d+)\.([-\d]+)$'
    match = re.match(pattern, portal_id)
    
    if match:
        return {
            'type': match.group(1),
            'number': match.group(2),
            'target_floor': match.group(3),
            'portal_id': portal_id
        }
    return None


def find_matching_portal(portal_info: Dict, target_floor: str, all_portals: List[Dict]) -> Optional[Dict]:
    """
    Bir portalın eşleşen karşılığını hedef katta bulur
    
    Örnek: Elev.8.1 (0. katta) -> Elev.8.0 (1. katta)
    
    Args:
        portal_info: parse_portal_name() çıktısı
        target_floor: Hedef kat numarası (string: '0', '-1', vb.)
        all_portals: Hedef kattaki tüm portal listesi
    
    Returns:
        Eşleşen portal bilgisi veya None
    """
    portal_type = portal_info['type']
    portal_number = portal_info['number']
    
    for portal in all_portals:
        parsed = parse_portal_name(portal['id'])
        if parsed:
            # Aynı tip ve numara, hedef kat eşleşiyor mu?
            if (parsed['type'] == portal_type and 
                parsed['number'] == portal_number and
                parsed['target_floor'] == target_floor):
                return portal
    
    return None


def get_floor_number(floor_name: str) -> int:
    """
    Kat isminden sayısal değer çıkarır
    
    Args:
        floor_name: 'Kat 0', 'Kat -1', '0', '-1' gibi
    
    Returns:
        Kat numarası (integer)
    """
    # 'Kat 0', 'Kat -1' formatından sayıyı çıkar
    if 'Kat' in floor_name or 'kat' in floor_name:
        parts = floor_name.split()
        if len(parts) >= 2:
            try:
                return int(parts[-1])
            except ValueError:
                pass
    
    # Direkt sayı ise
    try:
        return int(floor_name.strip())
    except ValueError:
        return 0


def find_available_portals_from_floor(current_floor: str, 
                                      current_graph,
                                      all_graphs: List,
                                      floor_names: List[str]) -> Dict[str, List[Dict]]:
    """
    Mevcut kattan hangi katlara portallar var?
    
    Args:
        current_floor: Mevcut kat ismi
        current_graph: Mevcut kat graph'ı
        all_graphs: Tüm katların graph'ları
        floor_names: Tüm kat isimleri
    
    Returns:
        {target_floor: [portal_list]} şeklinde dict
    """
    available_portals = {}
    
    # Mevcut kattaki tüm portalları incele
    for conn in current_graph.connections:
        if conn.get('type') == 'portal':
            portal_info = parse_portal_name(conn['id'])
            
            if portal_info:
                target_floor = portal_info['target_floor']
                
                if target_floor not in available_portals:
                    available_portals[target_floor] = []
                
                available_portals[target_floor].append({
                    'connection': conn,
                    'portal_info': portal_info
                })
    
    return available_portals


def plan_floor_transitions(start_floor: str, 
                          end_floor: str,
                          all_graphs: List,
                          floor_names: List[str],
                          start_point: Tuple[float, float] = None) -> List[Tuple[str, str, Dict]]:
    """
    Başlangıç katından hedef kata geçiş planı oluşturur
    Direkt portal yoksa aşamalı geçişleri planlar
    
    Args:
        start_floor: Başlangıç kat ismi
        end_floor: Hedef kat ismi
        all_graphs: Tüm katların graph'ları
        floor_names: Tüm kat isimleri
        start_point: Başlangıç noktası (x, y) - en yakın portalı seçmek için
    
    Returns:
        [(from_floor, to_floor, portal_info), ...] şeklinde geçiş listesi
    """
    from helpers.path_analysis import distance
    
    start_num = get_floor_number(start_floor)
    end_num = get_floor_number(end_floor)
    
    # Aynı kattaysak geçiş yok
    if start_num == end_num:
        return []
    
    # Hangi yöne gideceğiz? (yukarı/aşağı)
    direction = 1 if end_num > start_num else -1
    
    transitions = []
    current_floor_num = start_num
    current_point = start_point  # Mevcut konum (portal geçişlerinde güncellenir)
    
    # Kat kat ilerle
    while current_floor_num != end_num:
        # Mevcut katın graph'ını bul
        current_floor_name = None
        current_graph = None
        
        for i, fname in enumerate(floor_names):
            if get_floor_number(fname) == current_floor_num:
                current_floor_name = fname
                current_graph = all_graphs[i]
                break
        
        if not current_graph:
            print(f"Uyarı: {current_floor_num} numaralı kat bulunamadı")
            break
        
        # Bu kattan mevcut portalları bul
        available_portals = find_available_portals_from_floor(
            current_floor_name, current_graph, all_graphs, floor_names
        )
        
        # Hedef kata direkt portal var mı?
        target_floor_str = str(end_num)
        if target_floor_str in available_portals:
            # EN YAKIN portalı seç (başlangıç noktasına göre)
            best_portal = select_nearest_portal(
                available_portals[target_floor_str], 
                current_point
            )
            transitions.append((
                current_floor_name,
                next((fn for fn in floor_names if get_floor_number(fn) == end_num), None),
                best_portal
            ))
            break
        else:
            # Direkt portal yok, ara kata geç
            # Hedefe yaklaşan en iyi katı bul
            best_intermediate = None
            min_distance_to_target = float('inf')
            
            for target_floor_str, portals in available_portals.items():
                target_floor_num = int(target_floor_str)
                
                # Doğru yönde mi? (yukarı gidiyorsak yukarı, aşağı gidiyorsak aşağı)
                if (direction > 0 and target_floor_num > current_floor_num) or \
                   (direction < 0 and target_floor_num < current_floor_num):
                    
                    distance_to_target = abs(target_floor_num - end_num)
                    
                    if distance_to_target < min_distance_to_target:
                        min_distance_to_target = distance_to_target
                        # EN YAKIN portalı seç
                        best_portal = select_nearest_portal(portals, current_point)
                        best_intermediate = (target_floor_num, best_portal)
            
            if best_intermediate:
                intermediate_floor_num, portal = best_intermediate
                intermediate_floor_name = next(
                    (fn for fn in floor_names if get_floor_number(fn) == intermediate_floor_num), 
                    None
                )
                
                transitions.append((
                    current_floor_name,
                    intermediate_floor_name,
                    portal
                ))
                
                # Sonraki kat için mevcut konumu güncelle (portal'ın konumu)
                if portal and 'connection' in portal:
                    conn = portal['connection']
                    current_point = ((conn['x1'] + conn['x2']) / 2, (conn['y1'] + conn['y2']) / 2)
                
                current_floor_num = intermediate_floor_num
            else:
                print(f"Uyarı: {current_floor_name} katından ilerlenecek portal bulunamadı")
                break
    
    return transitions


def select_nearest_portal(portals: List[Dict], point: Tuple[float, float] = None) -> Dict:
    """
    Verilen noktaya en yakın portalı seçer
    
    Args:
        portals: Portal listesi
        point: Referans noktası (x, y)
    
    Returns:
        En yakın portal dict'i
    """
    if not portals:
        return None
    
    if point is None or len(portals) == 1:
        return portals[0]
    
    from helpers.path_analysis import distance
    
    min_dist = float('inf')
    nearest_portal = portals[0]
    
    for portal in portals:
        if 'connection' in portal:
            conn = portal['connection']
            # Portal'ın orta noktası
            portal_mid = ((conn['x1'] + conn['x2']) / 2, (conn['y1'] + conn['y2']) / 2)
            dist = distance(point, portal_mid)
            
            if dist < min_dist:
                min_dist = dist
                nearest_portal = portal
    
    return nearest_portal


def find_nearest_portal_to_point(point: Tuple[float, float],
                                 portals: List[Dict],
                                 graph) -> Optional[str]:
    """
    Bir noktaya en yakın portal connection ID'sini bulur
    
    Args:
        point: (x, y) koordinat
        portals: Portal bilgileri listesi
        graph: Graph nesnesi
    
    Returns:
        En yakın portal'ın connection ID'si
    """
    from helpers.path_analysis import distance
    
    min_dist = float('inf')
    nearest_portal_id = None
    
    for portal_data in portals:
        conn = portal_data['connection']
        
        # Portal'ın orta noktasını hesapla
        portal_mid_x = (conn['x1'] + conn['x2']) / 2
        portal_mid_y = (conn['y1'] + conn['y2']) / 2
        portal_point = (portal_mid_x, portal_mid_y)
        
        dist = distance(point, portal_point)
        
        if dist < min_dist:
            min_dist = dist
            nearest_portal_id = conn['id']
    
    return nearest_portal_id


def get_portal_connection_id(portal_id: str, graph) -> Optional[str]:
    """
    Portal ID'sinden connection ID'sini bulur
    
    Args:
        portal_id: Portal ismi (örn: 'Elev.8.1')
        graph: Graph nesnesi
    
    Returns:
        Connection ID'si
    """
    for conn in graph.connections:
        if conn['id'] == portal_id:
            return conn['id']
    return None




