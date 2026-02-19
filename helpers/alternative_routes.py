"""
Alternative Routes Generator - Farklı kriterlere göre alternatif rotalar üretir
"""
import heapq
import math
from typing import List, Dict, Optional, Tuple
from helpers.dijkstra import euclidean_distance


def calculate_turn_angle(conn1_id, conn2_id, graph) -> float:
    """
    İki ardışık connection arasındaki dönüş açısını hesaplar
    Kesişim noktasını bularak doğru yön vektörlerini hesaplar
    
    Returns:
        Açı (derece cinsinden), 0-180 arası
    """
    # Connection'ları al
    c1 = next((c for c in graph.connections if c['id'] == conn1_id), None)
    c2 = next((c for c in graph.connections if c['id'] == conn2_id), None)
    
    if not c1 or not c2:
        return 0
    
    # Kesişim noktasını bul
    c1_points = [(c1['x1'], c1['y1']), (c1['x2'], c1['y2'])]
    c2_points = [(c2['x1'], c2['y1']), (c2['x2'], c2['y2'])]
    
    intersection = None
    c1_other = None
    c2_other = None
    
    for i, p1 in enumerate(c1_points):
        for j, p2 in enumerate(c2_points):
            if abs(p1[0] - p2[0]) <= 0.5 and abs(p1[1] - p2[1]) <= 0.5:
                intersection = p1
                c1_other = c1_points[1 - i]
                c2_other = c2_points[1 - j]
                break
        if intersection:
            break
    
    if not intersection:
        # Kesişim bulunamadı - varsayılan
        return 0
    
    # v1: c1'in diğer ucundan kesişime gelen yön (giriş yönü)
    v1_x = intersection[0] - c1_other[0]
    v1_y = intersection[1] - c1_other[1]
    
    # v2: kesişimden c2'nin diğer ucuna giden yön (çıkış yönü)
    v2_x = c2_other[0] - intersection[0]
    v2_y = c2_other[1] - intersection[1]
    
    # Vektör büyüklükleri
    mag1 = math.sqrt(v1_x**2 + v1_y**2)
    mag2 = math.sqrt(v2_x**2 + v2_y**2)
    
    if mag1 == 0 or mag2 == 0:
        return 0
    
    # Dot product ile açıyı hesapla
    dot = v1_x * v2_x + v1_y * v2_y
    cos_theta = dot / (mag1 * mag2)
    cos_theta = max(min(cos_theta, 1), -1)
    angle = math.acos(cos_theta) * (180 / math.pi)
    
    return angle


def count_significant_turns(path: List[str], graph) -> int:
    """
    Rotadaki anlamlı dönüş sayısını hesaplar (30°-175° arası)
    """
    if len(path) < 2:
        return 0
    
    turn_count = 0
    for i in range(len(path) - 1):
        angle = calculate_turn_angle(path[i], path[i + 1], graph)
        
        # Anlamlı dönüş mü?
        if 30 < angle < 175:
            turn_count += 1
    
    return turn_count


def dijkstra_with_custom_cost(graph, start_id: str, goal_id: str, 
                              cost_type: str = "distance") -> Optional[Tuple[List[str], Dict]]:
    """
    Özelleştirilebilir maliyet fonksiyonu ile Dijkstra
    
    Args:
        graph: Graph nesnesi
        start_id: Başlangıç connection ID
        goal_id: Hedef connection ID
        cost_type: "distance" (en kısa), "turns" (en az dönüş), "balanced" (dengeli)
    
    Returns:
        (path, metrics) tuple: Rota ve metrikleri
    """
    # İlk connection'ın kendi uzunluğunu başlangıç maliyeti olarak say
    start_conn = next((c for c in graph.connections if c['id'] == start_id), None)
    if not start_conn:
        return None
    
    start_cost = euclidean_distance(
        (start_conn['x1'], start_conn['y1']),
        (start_conn['x2'], start_conn['y2'])
    )
    
    open_set = []
    heapq.heappush(open_set, (start_cost, start_id, []))  # (cost, conn_id, path_so_far)
    
    visited = set()
    best_cost = {conn['id']: float('inf') for conn in graph.connections}
    best_cost[start_id] = start_cost
    
    while open_set:
        current_cost, current_id, path_so_far = heapq.heappop(open_set)
        
        if current_id in visited:
            continue
        
        visited.add(current_id)
        current_path = path_so_far + [current_id]
        
        # Hedefe ulaştık mı?
        if current_id == goal_id:
            # Metrikleri hesapla
            total_distance = calculate_path_distance(current_path, graph)
            turn_count = count_significant_turns(current_path, graph)
            
            return current_path, {
                'total_distance': total_distance,
                'turn_count': turn_count,
                'step_count': len(current_path),
                'cost_type': cost_type
            }
        
        # Komşuları kontrol et
        for neighbor_id in graph.adjacency_list.get(current_id, []):
            if neighbor_id in visited:
                continue
            
            # Maliyet hesapla
            edge_cost = calculate_edge_cost(
                current_id, neighbor_id, current_path, graph, cost_type
            )
            
            tentative_cost = current_cost + edge_cost
            
            if tentative_cost < best_cost[neighbor_id]:
                best_cost[neighbor_id] = tentative_cost
                heapq.heappush(open_set, (tentative_cost, neighbor_id, current_path))
    
    return None


def find_intersection_point(conn1, conn2, tolerance=0.5):
    """
    İki connection'ın kesişim noktasını bul
    Returns: (intersection_point, conn1_other_end, conn2_other_end)
    """
    c1_points = [(conn1['x1'], conn1['y1']), (conn1['x2'], conn1['y2'])]
    c2_points = [(conn2['x1'], conn2['y1']), (conn2['x2'], conn2['y2'])]
    
    for i, p1 in enumerate(c1_points):
        for j, p2 in enumerate(c2_points):
            if abs(p1[0] - p2[0]) <= tolerance and abs(p1[1] - p2[1]) <= tolerance:
                # Kesişim noktası bulundu
                intersection = p1
                # conn1'in diğer ucu (kesişim olmayan)
                conn1_other = c1_points[1 - i]
                # conn2'nin diğer ucu (kesişim olmayan)
                conn2_other = c2_points[1 - j]
                return intersection, conn1_other, conn2_other
    
    # Kesişim bulunamadı - varsayılan olarak x2,y2 -> x1,y1 kullan
    return (conn1['x2'], conn1['y2']), (conn1['x1'], conn1['y1']), (conn2['x2'], conn2['y2'])


def calculate_edge_cost(from_id: str, to_id: str, path_so_far: List[str], 
                       graph, cost_type: str) -> float:
    """
    İki connection arasındaki geçiş maliyetini hesapla
    """
    # Temel mesafe maliyeti
    conn1 = next((c for c in graph.connections if c['id'] == from_id), None)
    conn2 = next((c for c in graph.connections if c['id'] == to_id), None)
    
    if not conn1 or not conn2:
        return float('inf')
    
    # Gerçek kesişim noktasını bul
    intersection, conn1_other, conn2_other = find_intersection_point(conn1, conn2)
    
    # Geçiş mesafesi: kesişim noktasında olduğumuz için 0 veya çok küçük
    # conn2'nin kendi uzunluğu (kesişimden diğer uca)
    conn2_length = euclidean_distance(intersection, conn2_other)
    
    # TOPLAM MESAFE = connection'un kendisi (kesişimden itibaren)
    distance_cost = conn2_length
    
    if cost_type == "distance":
        # SADECE MESAFE - en kısa yolu bul
        return distance_cost
    
    elif cost_type == "turns":
        # SADECE DÖNÜŞ - en az dönüşlü yolu bul
        turn_penalty = 0
        if len(path_so_far) > 0:
            angle = calculate_turn_angle(path_so_far[-1], to_id, graph)
            
            # Anlamlı dönüşlere çok yüksek ceza
            if 30 < angle < 175:
                turn_penalty = 1000  # Dönüşleri minimize et
        
        # Mesafe çok az önemli (sadece eşitlik durumunda)
        return turn_penalty + distance_cost * 0.01
    
    elif cost_type == "balanced":
        # Hem mesafe hem dönüş dengeli
        turn_penalty = 0
        if len(path_so_far) > 0:
            angle = calculate_turn_angle(path_so_far[-1], to_id, graph)
            
            if 30 < angle < 175:
                if angle > 80:
                    turn_penalty = 100
                else:
                    turn_penalty = 50
        
        return distance_cost + turn_penalty
    
    return distance_cost


def calculate_path_distance(path: List[str], graph) -> float:
    """Rota toplam mesafesini hesapla (tüm connection uzunlukları)"""
    if not path:
        return 0
    
    total = 0
    
    # İlk connection'ın kendi uzunluğu
    first_conn = next((c for c in graph.connections if c['id'] == path[0]), None)
    if first_conn:
        total += euclidean_distance(
            (first_conn['x1'], first_conn['y1']),
            (first_conn['x2'], first_conn['y2'])
        )
    
    # Her geçiş için: kesişim noktasından diğer uca olan mesafe
    for i in range(len(path) - 1):
        conn1 = next((c for c in graph.connections if c['id'] == path[i]), None)
        conn2 = next((c for c in graph.connections if c['id'] == path[i + 1]), None)
        
        if conn1 and conn2:
            # Gerçek kesişim noktasını bul
            intersection, conn1_other, conn2_other = find_intersection_point(conn1, conn2)
            
            # conn2'nin uzunluğu (kesişimden diğer uca)
            conn2_length = euclidean_distance(intersection, conn2_other)
            total += conn2_length
    
    return total


def generate_alternative_routes(graph, start_id: str, goal_id: str) -> Dict[str, Dict]:
    """
    Farklı kriterlere göre alternatif rotalar üretir
    
    Returns:
        {
            'shortest': {'path': [...], 'metrics': {...}},
            'least_turns': {'path': [...], 'metrics': {...}}
        }
    """
    routes = {}
    
    # 1. En kısa mesafe
    print("  → En kısa mesafe rotası hesaplanıyor...")
    result = dijkstra_with_custom_cost(graph, start_id, goal_id, "distance")
    if result:
        path, metrics = result
        routes['shortest'] = {
            'path': path,
            'metrics': metrics,
            'name': 'En Kısa Mesafe',
            'description': f"{metrics['total_distance']:.1f} birim, {metrics['turn_count']} dönüş"
        }
        print(f"    ✓ {metrics['total_distance']:.1f} birim, {metrics['turn_count']} dönüş")
    
    # 2. En az dönüş
    print("  → En az dönüş rotası hesaplanıyor...")
    result = dijkstra_with_custom_cost(graph, start_id, goal_id, "turns")
    if result:
        path, metrics = result
        routes['least_turns'] = {
            'path': path,
            'metrics': metrics,
            'name': 'En Az Dönüş',
            'description': f"{metrics['total_distance']:.1f} birim, {metrics['turn_count']} dönüş"
        }
        print(f"    ✓ {metrics['total_distance']:.1f} birim, {metrics['turn_count']} dönüş")
    
    return routes

