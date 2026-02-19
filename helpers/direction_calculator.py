"""
AR Uygulaması için Yön Hesaplama Modülü

Başlangıç noktasından ilk adımda hangi yöne gidileceğini hesaplar.
Kısa/anlamsız segmentleri filtreler ve ağırlıklı ortalama ile
anlamlı bir yön tahmini çıkarır.
"""

import math
from typing import List, Tuple, Optional, Dict


def calculate_initial_direction(path_points: List[Tuple[float, float]], 
                                max_segments: int = 5,
                                min_segment_length: float = 30.0) -> Optional[Dict]:
    """
    AR uygulaması için başlangıç yönünü hesaplar.
    
    Kısa/anlamsız segmentleri filtreler ve uzunluklarına göre ağırlıklandırarak
    anlamlı bir yön tahmini çıkarır.
    
    Args:
        path_points: Rota noktaları [(x1,y1), (x2,y2), ...]
        max_segments: Değerlendirilecek maksimum segment sayısı
        min_segment_length: Minimum segment uzunluğu (bundan kısa olanlar göz ardı edilir)
    
    Returns:
        {
            'angle_degrees': float,  # Derece cinsinden açı (SVG: 0=Sağ, saat yönünde)
            'angle_radians': float,  # Radyan cinsinden açı
            'compass_angle': float,  # Pusula açısı (0=Kuzey, saat yönünde)
            'direction_vector': (float, float),  # Normalize edilmiş yön vektörü
            'confidence': float,  # Güven skoru (0-1)
            'compass': str,  # Pusula yönü (Kuzey, Güneydoğu, vb.)
            'segments_used': int,  # Kaç segment kullanıldı
            'start_point': (float, float),  # Başlangıç noktası
            'arrow_end': (float, float)  # Ok uç noktası (görselleştirme için)
        }
    """
    if len(path_points) < 2:
        return None
    
    # Ağırlıklı vektör toplamı için
    weighted_dx = 0.0
    weighted_dy = 0.0
    total_weight = 0.0
    segments_used = 0
    segment_details = []
    
    # İlk N segment'i değerlendir
    for i in range(min(max_segments, len(path_points) - 1)):
        p1 = path_points[i]
        p2 = path_points[i + 1]
        
        # Segment vektörü
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        
        # Segment uzunluğu
        length = math.sqrt(dx*dx + dy*dy)
        
        segment_details.append({
            'index': i,
            'length': length,
            'start': p1,
            'end': p2,
            'skipped': length < min_segment_length
        })
        
        # Çok kısa segmentleri atla
        if length < min_segment_length:
            continue
        
        # Birim vektör × uzunluk (ağırlık)
        # Uzunluk^2 kullanarak uzun segmentlere daha fazla ağırlık ver
        weight = length * length
        
        # Mesafe azalma katsayısı: başlangıca yakın segmentler daha önemli
        distance_factor = 1.0 / (i + 1)  # İlk segment: 1.0, ikinci: 0.5, üçüncü: 0.33...
        weight *= distance_factor
        
        weighted_dx += dx * weight / length  # Normalize et ve ağırlıkla
        weighted_dy += dy * weight / length
        total_weight += weight
        segments_used += 1
    
    # Hiç geçerli segment yoksa, en kısa olanı bile kullan
    if segments_used == 0 and len(path_points) >= 2:
        dx = path_points[1][0] - path_points[0][0]
        dy = path_points[1][1] - path_points[0][1]
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            weighted_dx = dx / length
            weighted_dy = dy / length
            total_weight = 1.0
            segments_used = 1
    
    if total_weight == 0:
        return None
    
    # Ağırlıklı ortalama yön vektörü
    avg_dx = weighted_dx / total_weight
    avg_dy = weighted_dy / total_weight
    
    # Normalize et
    magnitude = math.sqrt(avg_dx*avg_dx + avg_dy*avg_dy)
    if magnitude > 0:
        avg_dx /= magnitude
        avg_dy /= magnitude
    
    # Açı hesapla (SVG koordinat sisteminde: Y aşağı doğru artar)
    # atan2(y, x) kullanarak 4 kadranı da doğru hesapla
    angle_rad = math.atan2(avg_dy, avg_dx)
    angle_deg = math.degrees(angle_rad)
    
    # SVG'den gerçek dünya koordinatlarına dönüşüm
    # SVG: 0° sağ, saat yönünde, Y aşağı
    # Pusula: 0° kuzey, saat yönünde
    compass_angle = (90 - angle_deg) % 360
    
    # Güven skoru: Vektörlerin ne kadar tutarlı olduğu
    confidence = min(1.0, magnitude * segments_used / max_segments)
    
    # Pusula yönü
    compass = get_compass_direction(compass_angle)
    
    # Başlangıç noktası
    start_point = path_points[0]
    
    # Ok uç noktası (görselleştirme için - 100 birim uzunluğunda)
    arrow_length = 100.0
    arrow_end = (
        start_point[0] + avg_dx * arrow_length,
        start_point[1] + avg_dy * arrow_length
    )
    
    return {
        'angle_degrees': angle_deg,
        'angle_radians': angle_rad,
        'compass_angle': compass_angle,
        'direction_vector': (avg_dx, avg_dy),
        'confidence': confidence,
        'compass': compass,
        'segments_used': segments_used,
        'start_point': start_point,
        'arrow_end': arrow_end,
        'segment_details': segment_details
    }


def calculate_direction_with_lookahead(path_points: List[Tuple[float, float]], 
                                       lookahead_distance: float = 200.0) -> Optional[Dict]:
    """
    Alternatif yaklaşım: Sabit bir mesafe ileriye bakarak yön hesapla.
    
    Bu yöntem, segment sayısı yerine toplam mesafeye göre çalışır.
    Kısa segmentler otomatik olarak birleştirilmiş olur.
    
    Args:
        path_points: Rota noktaları
        lookahead_distance: İleriye bakılacak mesafe (piksel/birim)
    """
    if len(path_points) < 2:
        return None
    
    start = path_points[0]
    accumulated_distance = 0.0
    target_point = path_points[-1]  # Varsayılan: son nokta
    
    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i + 1]
        
        segment_length = math.sqrt(
            (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2
        )
        
        if accumulated_distance + segment_length >= lookahead_distance:
            # Bu segment içinde interpolasyon yap
            remaining = lookahead_distance - accumulated_distance
            ratio = remaining / segment_length if segment_length > 0 else 0
            
            target_point = (
                p1[0] + (p2[0] - p1[0]) * ratio,
                p1[1] + (p2[1] - p1[1]) * ratio
            )
            break
        
        accumulated_distance += segment_length
        target_point = p2
    
    # Başlangıçtan hedef noktaya yön vektörü
    dx = target_point[0] - start[0]
    dy = target_point[1] - start[1]
    length = math.sqrt(dx*dx + dy*dy)
    
    if length == 0:
        return None
    
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    compass_angle = (90 - angle_deg) % 360
    
    # Ok uç noktası
    arrow_length = 100.0
    norm_dx = dx / length
    norm_dy = dy / length
    arrow_end = (
        start[0] + norm_dx * arrow_length,
        start[1] + norm_dy * arrow_length
    )
    
    return {
        'angle_degrees': angle_deg,
        'angle_radians': angle_rad,
        'compass_angle': compass_angle,
        'direction_vector': (norm_dx, norm_dy),
        'compass': get_compass_direction(compass_angle),
        'start_point': start,
        'target_point': target_point,
        'arrow_end': arrow_end,
        'lookahead_distance': min(accumulated_distance + length, lookahead_distance),
        'confidence': 1.0
    }


def get_compass_direction(angle: float) -> str:
    """Açıdan pusula yönünü döndürür"""
    directions = [
        (0, "Kuzey"), (22.5, "Kuzey-Kuzeydoğu"),
        (45, "Kuzeydoğu"), (67.5, "Doğu-Kuzeydoğu"),
        (90, "Doğu"), (112.5, "Doğu-Güneydoğu"),
        (135, "Güneydoğu"), (157.5, "Güney-Güneydoğu"),
        (180, "Güney"), (202.5, "Güney-Güneybatı"),
        (225, "Güneybatı"), (247.5, "Batı-Güneybatı"),
        (270, "Batı"), (292.5, "Batı-Kuzeybatı"),
        (315, "Kuzeybatı"), (337.5, "Kuzey-Kuzeybatı")
    ]
    
    # En yakın yönü bul
    angle = angle % 360
    for i, (deg, name) in enumerate(directions):
        next_deg = directions[(i + 1) % len(directions)][0]
        if next_deg == 0:
            next_deg = 360
        
        if deg <= angle < next_deg:
            # Hangisine daha yakın?
            if abs(angle - deg) <= abs(angle - next_deg):
                return name
            else:
                return directions[(i + 1) % len(directions)][1]
    
    return "Kuzey"


def extract_path_points_from_route(route_data: dict) -> List[Tuple[float, float]]:
    """
    Rota verisinden path noktalarını çıkarır.
    
    Args:
        route_data: generate_route_for_room_pair veya benzeri fonksiyondan dönen veri
    
    Returns:
        Path noktalarının listesi [(x1, y1), (x2, y2), ...]
    """
    path_points = []
    
    if 'path' not in route_data:
        return path_points
    
    for connection in route_data['path']:
        # Her connection'ın başlangıç ve bitiş noktasını al
        x1 = connection.get('x1', 0)
        y1 = connection.get('y1', 0)
        x2 = connection.get('x2', 0)
        y2 = connection.get('y2', 0)
        
        # İlk nokta ise veya önceki noktadan farklıysa ekle
        if not path_points:
            path_points.append((x1, y1))
        elif path_points[-1] != (x1, y1):
            # Bağlantı noktasını kontrol et
            if path_points[-1] == (x2, y2):
                # Ters yönde, x1,y1'i ekle
                pass
            else:
                path_points.append((x1, y1))
        
        # Bitiş noktasını ekle
        if not path_points or path_points[-1] != (x2, y2):
            path_points.append((x2, y2))
    
    return path_points





