"""
Akıllı Anchor (Referans Noktası) Seçim Sistemi

Gerçek bir iç mekanda yön tarifi yaparken insanların kullandığı
mantıksal referans noktası seçim kurallarını uygular.

Seçim Kriterleri:
1. Tip Önceliği - Bazı yerler doğal landmark'tır (asansör, tuvalet, eczane)
2. Mesafe - Çok uzak yerler referans olamaz
3. Görünürlük - Büyük ama çok büyük olmayan yerler ideal
4. Konum - Dönüş yönüne göre mantıklı konumda olmalı
5. Tanınırlık - Herkesin bildiği yerler tercih edilir
"""

from typing import Dict, List, Tuple, Optional
import math


# =============================================================================
# ANCHOR TİP KATEGORİLERİ VE ÖNCELİKLERİ
# =============================================================================

# Tier 1: Evrensel Landmark'lar (herkes bilir, her yerde aynı)
TIER_1_LANDMARKS = {
    'Elevator', 'Elev', 'Asansör', 'Lift',
    'Stairs', 'Staircase', 'Merdiven', 'Stair',
    'Escalator', 'Yürüyen Merdiven',
    'WC', 'Restroom', 'Toilet', 'Tuvalet', 'Lavabo',
    'Info', 'Information', 'Bilgi', 'Danışma',
    'Exit', 'Çıkış', 'Emergency Exit', 'Acil Çıkış',
    'Entrance', 'Giriş', 'Gate', 'Kapı'
}

# Tier 2: Ticari Landmark'lar (belirgin, tanınır)
TIER_2_COMMERCIAL = {
    'Pharmacy', 'Eczane', 'Medical', 'Sağlık',
    'Bank', 'Banka', 'ATM',
    'Food', 'Restaurant', 'Restoran', 'Cafe', 'Kafe', 'Coffee',
    'Shop', 'Store', 'Mağaza', 'Commercial', 'Dükkan',
    'Cinema', 'Sinema', 'Theater', 'Tiyatro',
    'Supermarket', 'Market', 'Grocery'
}

# Tier 3: Orta Öncelikli Yerler
TIER_3_SECONDARY = {
    'Social', 'Sosyal', 'Lounge', 'Bekleme',
    'Water', 'Fountain', 'Çeşme', 'Su',
    'Kiosk', 'Büfe',
    'Service', 'Servis'
}

# Tier 4: Anchor olarak KULLANILMAZ
TIER_4_EXCLUDED = {
    'Parking', 'Otopark', 'Car Park',
    'Building', 'Bina', 'Structure',
    'Walking', 'Yürüme', 'Corridor', 'Koridor', 'Hallway',
    'Storage', 'Depo',
    'Technical', 'Teknik', 'Mechanical',
    'Void', 'Boşluk'
}


def get_anchor_tier(room_type: str) -> int:
    """
    Oda tipinin anchor öncelik seviyesini belirler
    
    Returns:
        1: En iyi (evrensel landmark)
        2: İyi (ticari)
        3: Orta
        4: Kullanılmaz
        5: Bilinmeyen (varsayılan orta-düşük)
    """
    room_type_upper = room_type.upper() if room_type else ""
    room_type_lower = room_type.lower() if room_type else ""
    
    # Tier 1 kontrolü
    for keyword in TIER_1_LANDMARKS:
        if keyword.upper() in room_type_upper or keyword.lower() in room_type_lower:
            return 1
    
    # Tier 2 kontrolü
    for keyword in TIER_2_COMMERCIAL:
        if keyword.upper() in room_type_upper or keyword.lower() in room_type_lower:
            return 2
    
    # Tier 3 kontrolü
    for keyword in TIER_3_SECONDARY:
        if keyword.upper() in room_type_upper or keyword.lower() in room_type_lower:
            return 3
    
    # Tier 4 kontrolü (hariç tutulanlar)
    for keyword in TIER_4_EXCLUDED:
        if keyword.upper() in room_type_upper or keyword.lower() in room_type_lower:
            return 4
    
    # Bilinmeyen tip
    return 5


def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """İki nokta arası Öklid mesafesi"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def get_distance_score(distance: float, pixel_to_meter: float = 0.1) -> float:
    """
    Mesafeye göre skor çarpanı hesaplar
    
    Anchor dönüş noktasına YAKIN olmalı - uzaktaki anchor işe yaramaz!
    
    Mesafe Kuralları:
    - 0-5m: Mükemmel, tam yanında (1.0)
    - 5-10m: Çok iyi, hemen yakınında (0.9)
    - 10-20m: İyi, görüş alanında (0.7)
    - 20-35m: Orta, biraz uzak (0.4)
    - 35-50m: Zayıf (0.2)
    - 50m+: Çok uzak, son çare (0.05)
    """
    distance_meters = distance * pixel_to_meter
    
    if distance_meters <= 5:
        return 1.0  # Tam yanında
    elif distance_meters <= 10:
        return 0.9  # Çok yakın
    elif distance_meters <= 20:
        return 0.7  # İyi mesafe
    elif distance_meters <= 35:
        return 0.4  # Biraz uzak
    elif distance_meters <= 50:
        return 0.2  # Zayıf
    else:
        return 0.05  # Çok uzak - neredeyse kullanılmaz


def get_size_score(area: float) -> float:
    """
    Boyuta göre skor çarpanı hesaplar
    
    İdeal anchor: Küçük-orta boyutlu, kolayca tanınabilir yerler
    Kötü anchor: Çok küçük (görülmez) veya çok büyük (belirsiz)
    
    Boyut Kuralları:
    - Çok küçük (<300): Görülmesi çok zor (0.3)
    - Küçük (300-1500): İdeal mağaza boyutu (1.0)
    - Orta (1500-5000): İyi (0.9)
    - Büyük (5000-15000): Belirsiz olabilir (0.5)
    - Çok büyük (15000-50000): Genel alan gibi (0.2)
    - Dev (>50000): Otopark, bina vs (0.05)
    """
    if area < 300:
        return 0.3  # Çok küçük, görülmez
    elif area < 1500:
        return 1.0  # İdeal boyut
    elif area < 5000:
        return 0.9  # İyi
    elif area < 15000:
        return 0.5  # Belirsiz
    elif area < 50000:
        return 0.2  # Çok büyük
    else:
        return 0.05  # Dev alanlar (otopark)


def get_tier_base_score(tier: int) -> float:
    """Tier'a göre temel skor"""
    scores = {
        1: 100,  # Evrensel landmark
        2: 80,   # Ticari
        3: 50,   # Orta
        4: 0,    # Kullanılmaz
        5: 30    # Bilinmeyen
    }
    return scores.get(tier, 30)


def get_direction_context(turn_point: Tuple[float, float], 
                          anchor_center: Tuple[float, float],
                          incoming_vector: Tuple[float, float] = None) -> str:
    """
    Anchor'ın geliş yönüne göre konumunu belirler
    
    Args:
        turn_point: Dönüş noktası
        anchor_center: Anchor merkezi
        incoming_vector: Geliş yönü vektörü (prev_point -> turn_point)
    
    Returns:
        'önünde', 'sağında', 'solunda', 'yanında', 'köşesinde'
    """
    # Anchor'a olan vektör
    dx = anchor_center[0] - turn_point[0]
    dy = anchor_center[1] - turn_point[1]
    
    distance = math.sqrt(dx*dx + dy*dy)
    if distance < 30:  # Çok yakınsa
        return 'yanında'
    
    # incoming_vector varsa, geliş yönüne göre relatif konum hesapla
    if incoming_vector and (incoming_vector[0] != 0 or incoming_vector[1] != 0):
        # Geliş yönünü normalize et
        in_mag = math.sqrt(incoming_vector[0]**2 + incoming_vector[1]**2)
        in_norm = (incoming_vector[0] / in_mag, incoming_vector[1] / in_mag)
        
        # Anchor yönünü normalize et
        anchor_mag = math.sqrt(dx*dx + dy*dy)
        anchor_norm = (dx / anchor_mag, dy / anchor_mag)
        
        # Dot product = cos(açı) - önde mi arkada mı?
        dot = in_norm[0] * anchor_norm[0] + in_norm[1] * anchor_norm[1]
        
        # Cross product = hangi tarafta (sağ/sol)?
        cross = in_norm[0] * anchor_norm[1] - in_norm[1] * anchor_norm[0]
        
        # dot > 0.7: önde (~45 derece)
        # dot < -0.3: arkada (ama arkada seçmemeliyiz zaten)
        # cross > 0: sağda (SVG koordinat sisteminde)
        # cross < 0: solda
        
        if dot > 0.7:
            return 'önünde'
        elif dot > 0.3:
            # Önde ama biraz yanda
            if cross > 0:
                return 'sağ önünde'
            else:
                return 'sol önünde'
        elif dot > -0.3:
            # Tamamen yanda
            if cross > 0:
                return 'sağında'
            else:
                return 'solunda'
        else:
            # Köşede veya arkada (ama arkada seçmemeliyiz)
            return 'köşesinde'
    
    # Fallback: Basit yön belirleme (incoming yoksa)
    angle = math.atan2(dy, dx) * 180 / math.pi
    
    if -45 <= angle < 45:
        return 'sağında'
    elif 45 <= angle < 135:
        return 'önünde'
    elif -135 <= angle < -45:
        return 'arkasında'
    else:
        return 'solunda'


def is_in_view_cone(turn_point: Tuple[float, float],
                    anchor_center: Tuple[float, float],
                    incoming_vector: Tuple[float, float] = None,
                    outgoing_vector: Tuple[float, float] = None) -> Tuple[bool, float]:
    """
    Anchor'ın görüş açısı içinde olup olmadığını kontrol eder.
    
    Mantık:
    - Kullanıcı dönüş noktasına gelirken baktığı yön = incoming_vector
    - Dönüşten sonra bakacağı yön = outgoing_vector
    - Anchor bu iki yönün birleşiminde olmalı (arkada kalmamalı)
    
    Önemli: SVG koordinat sisteminde Y aşağı doğru pozitif!
    
    Returns:
        (görüş_içinde_mi, görüş_skoru)
        görüş_skoru: 1.0 = tam önde/yanda, 0.0 = arkada
    """
    if incoming_vector is None and outgoing_vector is None:
        return True, 0.8  # Yön bilgisi yoksa kabul et ama düşük skor
    
    # Anchor'a olan vektör
    anchor_vec = (
        anchor_center[0] - turn_point[0],
        anchor_center[1] - turn_point[1]
    )
    
    # Vektör uzunluğu
    anchor_mag = math.sqrt(anchor_vec[0]**2 + anchor_vec[1]**2)
    if anchor_mag < 1.0:
        return True, 1.0  # Çok yakın, görüş açısı önemsiz
    
    # Normalize
    anchor_norm = (anchor_vec[0] / anchor_mag, anchor_vec[1] / anchor_mag)
    
    in_score = 0.0
    out_score = 0.0
    
    # Geliş yönü ile açı kontrolü
    # Anchor, geliş yönünün önünde veya yanında mı?
    if incoming_vector and (incoming_vector[0] != 0 or incoming_vector[1] != 0):
        in_mag = math.sqrt(incoming_vector[0]**2 + incoming_vector[1]**2)
        in_norm = (incoming_vector[0] / in_mag, incoming_vector[1] / in_mag)
        
        # Dot product: pozitif = ilerleme yönünde, negatif = arkada
        dot_in = anchor_norm[0] * in_norm[0] + anchor_norm[1] * in_norm[1]
        
        if dot_in >= -0.2:  # -0.2 = yaklaşık 100 derece tolerans (yanlara da bak)
            in_score = (dot_in + 1) / 2  # 0-1 arası normalize
    
    # Gidiş yönü ile açı kontrolü  
    # Anchor, gidiş yönünün önünde veya yanında mı?
    if outgoing_vector and (outgoing_vector[0] != 0 or outgoing_vector[1] != 0):
        out_mag = math.sqrt(outgoing_vector[0]**2 + outgoing_vector[1]**2)
        out_norm = (outgoing_vector[0] / out_mag, outgoing_vector[1] / out_mag)
        
        dot_out = anchor_norm[0] * out_norm[0] + anchor_norm[1] * out_norm[1]
        
        if dot_out >= -0.2:
            out_score = (dot_out + 1) / 2
    
    # En iyi skoru al - anchor ya geliş ya da gidiş yönünde görünür olmalı
    best_score = max(in_score, out_score)
    
    # Anchor en az bir yönde görünür olmalı
    # Eğer her iki yönde de negatif skor varsa, anchor arkada demektir
    is_visible = best_score > 0.3  # 0.3 = yaklaşık 70 derece dışında değilse görünür
    
    return is_visible, best_score


def calculate_anchor_score(room_type: str, 
                           room_id: str,
                           area: float, 
                           distance: float,
                           turn_point: Tuple[float, float],
                           room_center: Tuple[float, float],
                           pixel_to_meter: float = 0.1,
                           incoming_vector: Tuple[float, float] = None,
                           outgoing_vector: Tuple[float, float] = None) -> Dict:
    """
    Bir odanın anchor skoru hesaplar
    
    Returns:
        {
            'score': float,
            'tier': int,
            'distance_score': float,
            'size_score': float,
            'context': str,
            'usable': bool,
            'reason': str
        }
    """
    tier = get_anchor_tier(room_type)
    
    # Tier 4 (Parking, Walking vs.) kullanılmaz
    if tier == 4:
        return {
            'score': 0,
            'tier': tier,
            'distance_score': 0,
            'size_score': 0,
            'context': '',
            'usable': False,
            'reason': f'{room_type} tipi anchor olarak kullanılamaz'
        }
    
    # Mesafe skoru (artık hiçbir zaman 0 değil, en düşük 0.1)
    distance_score = get_distance_score(distance, pixel_to_meter)
    
    # Not: Artık çok uzak anchor'lar da düşük skorla kullanılabilir
    # Çünkü büyük mekanlarda (hastane, AVM) 100m+ mesafeler normal
    if distance_score < 0.1:  # Teorik olarak asla olmayacak
        return {
            'score': 0,
            'tier': tier,
            'distance_score': 0,
            'size_score': 0,
            'context': '',
            'usable': False,
            'reason': f'Çok uzak ({distance * pixel_to_meter:.1f}m)'
        }
    
    # Boyut skoru
    size_score = get_size_score(area)
    
    # GÖRÜŞ AÇISI KONTROLÜ - Anchor arkada kalmamalı!
    is_visible, view_score = is_in_view_cone(
        turn_point=turn_point,
        anchor_center=room_center,
        incoming_vector=incoming_vector,
        outgoing_vector=outgoing_vector
    )
    
    # Görüş açısı dışındaysa (arkada) kullanma
    if not is_visible:
        return {
            'score': 0,
            'tier': tier,
            'distance_score': distance_score,
            'size_score': size_score,
            'view_score': view_score,
            'context': 'arkada',
            'usable': False,
            'reason': 'Görüş açısı dışında (arkada)'
        }
    
    # Tier bazlı temel skor
    base_score = get_tier_base_score(tier)
    
    # Toplam skor hesapla (görüş skoru da dahil)
    total_score = base_score * distance_score * size_score * view_score
    
    # Konum bağlamı
    context = get_direction_context(turn_point, room_center, incoming_vector)
    
    # Bonus: Tier 1 (asansör, tuvalet) için mesafe daha az önemli
    if tier == 1 and distance_score >= 0.2:
        total_score *= 1.3  # %30 bonus
    
    # Bonus: Çok yakın ve orta boyut ideal
    if distance_score == 1.0 and size_score >= 0.9:
        total_score *= 1.2  # %20 bonus
    
    # Bonus: Tam önde olan anchor'lar daha iyi
    if view_score > 0.8:
        total_score *= 1.2  # %20 bonus
    
    return {
        'score': total_score,
        'tier': tier,
        'distance_score': distance_score,
        'size_score': size_score,
        'view_score': view_score,
        'context': context,
        'usable': True,
        'reason': 'OK'
    }


def find_best_anchor(turn_point: Tuple[float, float],
                     room_areas: Dict[str, List[Dict]],
                     max_distance: float = 1000,  # Büyük haritalar için artırıldı
                     pixel_to_meter: float = 0.1,
                     excluded_ids: List[str] = None,
                     debug: bool = False,
                     incoming_vector: Tuple[float, float] = None,
                     outgoing_vector: Tuple[float, float] = None) -> Optional[Tuple]:
    """
    Dönüş noktası için en iyi anchor'ı bulur
    
    Args:
        turn_point: Dönüş noktası koordinatları (x, y)
        room_areas: Oda bilgileri {'RoomType': [{'id', 'center', 'area'}, ...]}
        max_distance: Maksimum arama mesafesi (piksel)
        pixel_to_meter: Piksel-metre dönüşüm oranı
        excluded_ids: Hariç tutulacak oda ID'leri
        debug: Debug mesajları göster
        incoming_vector: Geliş yönü vektörü (görüş açısı için)
        outgoing_vector: Gidiş yönü vektörü (görüş açısı için)
    
    Returns:
        (room_type, room_id, area, distance, score_info) veya None
    """
    if excluded_ids is None:
        excluded_ids = []
    
    candidates = []
    checked_count = 0
    excluded_by_type = 0
    excluded_by_distance = 0
    excluded_by_view = 0  # Görüş açısı dışında kalanlar
    
    for room_type, rooms in room_areas.items():
        # Walking ve Building gibi tipleri atla (Tier 4)
        tier = get_anchor_tier(room_type)
        if tier == 4:
            excluded_by_type += len(rooms)
            continue
            
        for room in rooms:
            room_id = room.get('id', '')
            checked_count += 1
            
            # Hariç tutulanları atla
            if room_id in excluded_ids:
                continue
            
            center = room.get('center')
            if not center:
                continue
            
            # Mesafe hesapla
            distance = calculate_distance(turn_point, center)
            
            # Maksimum mesafe kontrolü
            if distance > max_distance:
                excluded_by_distance += 1
                continue
            
            area = room.get('area', 0)
            
            # Skor hesapla (görüş açısı dahil)
            score_info = calculate_anchor_score(
                room_type=room_type,
                room_id=room_id,
                area=area,
                distance=distance,
                turn_point=turn_point,
                room_center=center,
                pixel_to_meter=pixel_to_meter,
                incoming_vector=incoming_vector,
                outgoing_vector=outgoing_vector
            )
            
            if score_info['usable']:
                candidates.append({
                    'room_type': room_type,
                    'room_id': room_id,
                    'area': area,
                    'distance': distance,
                    'center': center,
                    'score_info': score_info
                })
            elif 'arkada' in score_info.get('reason', '').lower() or 'görüş' in score_info.get('reason', '').lower():
                excluded_by_view += 1
    
    if debug:
        print(f"    [Anchor Debug] Kontrol: {checked_count}, Tip hariç: {excluded_by_type}, Mesafe hariç: {excluded_by_distance}, Görüş hariç: {excluded_by_view}, Aday: {len(candidates)}")
    
    if not candidates:
        if debug:
            print(f"    [Anchor Debug] Hiç aday bulunamadı! turn_point={turn_point}")
        return None
    
    # En yüksek skorlu olanı seç
    candidates.sort(key=lambda x: x['score_info']['score'], reverse=True)
    best = candidates[0]
    
    if debug:
        print(f"    [Anchor Debug] Seçilen: {best['room_type']}/{best['room_id']} skor={best['score_info']['score']:.1f}")
    
    return (
        best['room_type'],
        best['room_id'],
        best['area'],
        best['distance'],
        best['score_info']
    )


def find_nearest_largest_room_smart(turn_point: Tuple[float, float],
                                     room_areas: Dict[str, List[Dict]],
                                     max_distance: float = 500,
                                     pixel_to_meter: float = 0.1) -> Optional[Tuple]:
    """
    Eski fonksiyonla uyumlu wrapper - akıllı anchor seçimi yapar
    
    Returns:
        (room_type, room_id, area, distance) veya None
    """
    result = find_best_anchor(
        turn_point=turn_point,
        room_areas=room_areas,
        max_distance=max_distance,
        pixel_to_meter=pixel_to_meter
    )
    
    if result:
        room_type, room_id, area, distance, score_info = result
        return (room_type, room_id, area, distance)
    
    return None


def get_anchor_description(anchor_info: Tuple, 
                           turn_direction: str = None,
                           pixel_to_meter: float = 0.1) -> str:
    """
    Anchor için insan dostu açıklama oluşturur
    
    Args:
        anchor_info: (room_type, room_id, area, distance) veya 5 elemanlı tuple
        turn_direction: 'sola' veya 'sağa'
    
    Returns:
        "Eczanenin yanında sola dönün" gibi açıklama
    """
    if not anchor_info:
        return ""
    
    room_type = anchor_info[0]
    room_id = anchor_info[1]
    distance = anchor_info[3]
    
    distance_meters = distance * pixel_to_meter
    
    # Mesafeye göre konum ifadesi
    if distance_meters <= 10:
        location = "yanında"
    elif distance_meters <= 25:
        location = "yakınında"
    else:
        location = "ilerisinde"
    
    # Oda tipi Türkçeleştirme
    type_names = {
        'Shop': 'mağaza',
        'Commercial': 'dükkan',
        'Food': 'restoran',
        'Medical': 'sağlık birimi',
        'Pharmacy': 'eczane',
        'WC': 'tuvalet',
        'Restroom': 'tuvalet',
        'Elevator': 'asansör',
        'Stairs': 'merdiven',
        'Escalator': 'yürüyen merdiven',
        'Info': 'danışma',
        'Bank': 'banka',
        'ATM': 'ATM',
        'Social': 'sosyal alan',
        'Water': 'çeşme',
        'Cinema': 'sinema',
        'Cafe': 'kafe'
    }
    
    type_name = type_names.get(room_type, room_type)
    
    # Açıklama oluştur
    if turn_direction:
        return f"{type_name.capitalize()} ({room_id}) {location} {turn_direction} dönün"
    else:
        return f"{type_name.capitalize()} ({room_id}) {location}"


# =============================================================================
# PORTAL/YAPISAL ELEMENT ANCHOR SEÇİMİ
# =============================================================================

def find_structural_anchor(turn_point: Tuple[float, float],
                           graph,
                           max_distance: float = 100) -> Optional[Dict]:
    """
    Dönüş noktası yakınındaki yapısal elementleri (asansör, merdiven) arar
    
    Bu fonksiyon, Rooms grubunda olmayan ama Portals grubundaki
    asansör/merdiven gibi yapısal elementleri anchor olarak kullanır.
    """
    if not hasattr(graph, 'connections'):
        return None
    
    best_structural = None
    min_distance = float('inf')
    
    for conn in graph.connections:
        conn_type = conn.get('type', '')
        conn_id = conn.get('id', '')
        
        # Portal (asansör/merdiven) mi kontrol et
        if conn_type == 'portal' or 'Elev' in conn_id or 'Stairs' in conn_id or 'Stair' in conn_id:
            # Connection'ın merkez noktası
            mid_x = (conn['x1'] + conn['x2']) / 2
            mid_y = (conn['y1'] + conn['y2']) / 2
            
            distance = calculate_distance(turn_point, (mid_x, mid_y))
            
            if distance < max_distance and distance < min_distance:
                min_distance = distance
                
                # Tip belirleme
                if 'Elev' in conn_id:
                    struct_type = 'Asansör'
                elif 'Stair' in conn_id or 'Stairs' in conn_id:
                    struct_type = 'Merdiven'
                elif 'Escal' in conn_id:
                    struct_type = 'Yürüyen Merdiven'
                else:
                    struct_type = 'Geçiş'
                
                best_structural = {
                    'type': struct_type,
                    'id': conn_id,
                    'distance': distance,
                    'center': (mid_x, mid_y)
                }
    
    return best_structural


def find_best_anchor_with_structural(turn_point: Tuple[float, float],
                                      room_areas: Dict[str, List[Dict]],
                                      graph = None,
                                      max_distance: float = 1000,  # Büyük haritalar için artırıldı
                                      pixel_to_meter: float = 0.1,
                                      debug: bool = False,
                                      incoming_vector: Tuple[float, float] = None,
                                      outgoing_vector: Tuple[float, float] = None) -> Optional[Tuple]:
    """
    Hem odaları hem yapısal elementleri değerlendirerek en iyi anchor'ı bulur
    Görüş açısı kontrolü yapar - arkada kalan anchor'ları seçmez
    """
    # Önce odalardan anchor bul (görüş açısı dahil)
    room_anchor = find_best_anchor(
        turn_point=turn_point,
        room_areas=room_areas,
        max_distance=max_distance,
        pixel_to_meter=pixel_to_meter,
        debug=debug,
        incoming_vector=incoming_vector,
        outgoing_vector=outgoing_vector
    )
    
    # Yapısal element ara (asansör, merdiven)
    structural_anchor = None
    if graph:
        structural_anchor = find_structural_anchor(
            turn_point=turn_point,
            graph=graph,
            max_distance=max_distance * 0.5  # Yapısal elementler için daha kısa mesafe
        )
    
    # Karşılaştır ve en iyisini seç
    # ÖNCELİK: Odalar > Yapısal elementler (çünkü odalar daha tanınır)
    # Yapısal elementler sadece çok yakın (5m) ve oda yoksa kullanılsın
    
    if room_anchor:
        # Oda anchor varsa onu tercih et
        # Yapısal element sadece ÇOK yakınsa (5m içinde) ve oda uzaksa tercih edilir
        if structural_anchor:
            struct_dist_m = structural_anchor['distance'] * pixel_to_meter
            room_dist_m = room_anchor[3] * pixel_to_meter
            
            # Yapısal element 5m içinde VE odadan en az 3x daha yakınsa
            if struct_dist_m <= 5 and struct_dist_m < room_dist_m / 3:
                return (
                    structural_anchor['type'],
                    structural_anchor['id'],
                    0,
                    structural_anchor['distance'],
                    {'tier': 1, 'score': 150}
                )
        
        # Aksi halde oda anchor'ı kullan
        return room_anchor
    
    elif structural_anchor:
        # Oda bulunamadı, yapısal element kullan
        return (
            structural_anchor['type'],
            structural_anchor['id'],
            0,
            structural_anchor['distance'],
            {'tier': 1, 'score': 100}
        )
    
    return None

