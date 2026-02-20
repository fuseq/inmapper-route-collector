"""
Route Directions Generator - Metrik Tabanlı:
Teknik, mesafe odaklı (GPS benzeri) rota tarifleri oluşturur
"""
import math
from typing import List, Tuple, Dict, Optional
from helpers.path_analysis import distance


class RouteStep:
    """Tek bir rota adımını temsil eder"""
    
    def __init__(self, step_number: int, action: str, distance_meters: float, 
                 cumulative_distance: float, description: str, 
                 landmark: Optional[str] = None, direction: Optional[str] = None,
                 alt_landmarks: Optional[List[str]] = None):
        self.step_number = step_number
        self.action = action  # "START", "CONTINUE", "TURN_LEFT", "TURN_RIGHT", "ARRIVE"
        self.distance_meters = distance_meters
        self.cumulative_distance = cumulative_distance
        self.description = description
        self.landmark = landmark
        self.direction = direction
        self.alt_landmarks = alt_landmarks or []
    
    def __str__(self):
        return f"{self.step_number}. {self.description}"
    
    def to_dict(self):
        """JSON formatına dönüştür"""
        d = {
            "step_number": self.step_number,
            "action": self.action,
            "distance_meters": round(self.distance_meters, 1),
            "cumulative_distance": round(self.cumulative_distance, 1),
            "description": self.description,
            "landmark": self.landmark,
            "direction": self.direction
        }
        if self.alt_landmarks:
            d["alt_landmarks"] = self.alt_landmarks
        return d


class MetricRouteGenerator:
    """Metrik tabanlı rota tarifleri oluşturur (GPS benzeri)"""
    
    def __init__(self, pixel_to_meter_ratio: float = 0.1):
        """
        Args:
            pixel_to_meter_ratio: SVG piksel birimlerini metreye çeviren oran
        """
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
        self.steps = []
    
    def generate_directions(self, path_points: List[Tuple[float, float]], 
                          turns: List[Dict], 
                          start_location: str = "Başlangıç noktası",
                          end_location: str = "Hedef nokta",
                          start_room_center: Optional[Tuple[float, float]] = None,
                          path_conn_ids: List[str] = None,
                          graph = None) -> List[RouteStep]:
        """
        Path points ve turn bilgilerinden adım adım yol tarifi oluşturur
        
        Args:
            path_points: Rotadaki tüm noktalar [(x, y), ...]
            turns: Dönüş noktaları bilgisi [{'point': (x,y), 'direction': 'sağa', 'anchor': ...}, ...]
            start_location: Başlangıç konumu açıklaması
            end_location: Hedef konumu açıklaması
            start_room_center: Başlangıç odasının merkez koordinatları (mağaza tarafı belirlemek için)
            path_conn_ids: Rota connection ID'leri (kapı segmentlerini atlamak için)
            graph: Graph nesnesi (connection tiplerini öğrenmek için)
        
        Returns:
            RouteStep listesi
        """
        self.steps = []
        cumulative_distance = 0
        step_number = 1
        
        # Toplam mesafeyi hesapla
        total_distance = self._calculate_total_distance(path_points)
        
        # 1. Başlangıç adımı
        first_turn_idx = self._find_first_turn_index(path_points, turns)
        first_segment_distance = self._calculate_segment_distance(path_points, 0, first_turn_idx)
        
        # Başlangıç açıklaması - mağaza tarafını belirle
        store_side = self._compute_store_side(path_points, start_room_center,
                                              path_conn_ids, graph)
        
        if store_side == 'arka':
            description = f"{start_location} noktasından çıkıp düz ilerleyin"
        elif store_side:
            side_text = "solunuzda" if store_side == "sol" else "sağınızda"
            description = f"{start_location} {side_text} kalacak şekilde ilerleyin"
        else:
            description = f"{start_location} noktasından çıkıp düz ilerleyin"
        
        start_step = RouteStep(
            step_number=step_number,
            action="START",
            distance_meters=first_segment_distance,
            cumulative_distance=cumulative_distance,
            description=description,
            direction=store_side
        )
        self.steps.append(start_step)
        cumulative_distance += first_segment_distance
        step_number += 1
        
        # 2. Her dönüş/kırılım için bir adım oluştur
        for i, turn in enumerate(turns):
            turn_point = turn['point']
            turn_direction = turn['direction']
            turn_type = turn.get('turn_type', 'turn')
            anchor = turn.get('anchor')
            anchor_side = turn.get('anchor_side')
            
            # Bu noktadan sonraki mesafeyi hesapla
            turn_index = self._find_turn_index_in_path(path_points, turn_point)
            _anc = f"{anchor[0]}-{anchor[1]}" if anchor else "?"
            print(f"    [DEBUG] i={i} type={turn_type} dir={turn_direction} anchor={_anc} "
                  f"turn_index={turn_index} path_len={len(path_points)}")
            
            if turn_index is not None and turn_index < len(path_points) - 1:
                # Bir sonraki noktaya veya hedefe olan mesafe
                if i < len(turns) - 1:
                    next_turn_point = turns[i + 1]['point']
                    next_turn_index = self._find_turn_index_in_path(path_points, next_turn_point)
                    segment_distance = self._calculate_segment_distance(path_points, turn_index, next_turn_index)
                else:
                    segment_distance = self._calculate_segment_distance(path_points, turn_index, len(path_points) - 1)
                
                # Landmark bilgisini oluştur
                landmark_text = None
                alt_landmarks = None
                if anchor:
                    room_type, room_id, area, dist = anchor[:4]
                    landmark_text = f"{room_type} - {room_id}"
                    # Collect alternative anchors if available
                    alt_anchors_list = anchor[4] if len(anchor) > 4 else []
                    alt_landmark_texts = []
                    for alt in alt_anchors_list:
                        alt_landmark_texts.append(f"{alt[0]} - {alt[1]}")
                    alt_landmarks = alt_landmark_texts if alt_landmark_texts else None
                
                if turn_type == 'veer':
                    # Zigzag → hafif yönelme talimatı
                    description = self._create_veer_description(
                        turn_direction, landmark_text, anchor_side, segment_distance
                    )
                    action = "VEER"
                    step_direction = turn_direction
                elif turn_type == 'bend':
                    # Koridor kırılımı → landmark referansı ile devam talimatı
                    description = self._create_pass_by_description(
                        landmark_text, anchor_side, segment_distance
                    )
                    action = "PASS_BY"
                    step_direction = anchor_side
                else:
                    # Gerçek dönüş → dönüş talimatı
                    description = self._create_turn_description(
                        turn_direction, segment_distance, landmark_text
                    )
                    action = f"TURN_{'LEFT' if turn_direction == 'sola' else 'RIGHT'}"
                    step_direction = turn_direction
                
                step = RouteStep(
                    step_number=step_number,
                    action=action,
                    distance_meters=segment_distance,
                    cumulative_distance=cumulative_distance,
                    description=description,
                    landmark=landmark_text,
                    direction=step_direction,
                    alt_landmarks=alt_landmarks
                )
                self.steps.append(step)
                cumulative_distance += segment_distance
                step_number += 1
        
        # 3. Varış adımı
        arrival_step = RouteStep(
            step_number=step_number,
            action="ARRIVE",
            distance_meters=0,
            cumulative_distance=cumulative_distance,
            description=f"{end_location} konumuna ulaştınız",
        )
        self.steps.append(arrival_step)
        
        return self.steps
    
    def _compute_store_side(self, path_points: List[Tuple[float, float]], 
                           start_room_center: Optional[Tuple[float, float]] = None,
                           path_conn_ids: List[str] = None,
                           graph = None) -> Optional[str]:
        """
        Başlangıç odasının koridordaki yürüme yönüne göre hangi tarafta
        kalacağını hesaplar.
        
        Kapı segmentlerini atlayarak ilk KORİDOR segmentinin yönüne bakılır.
        Oda tarafı belirlemek için oda merkezi yerine KAPI'nın oda tarafındaki
        ucu kullanılır — bu nokta her zaman koridorun doğru tarafındadır ve
        büyük/uzak polygon merkezlerinden etkilenmez.
        
        SVG koordinat sisteminde (Y aşağı doğru artar, top-down harita):
        - cross > 0 → mağaza sağda
        - cross < 0 → mağaza solda
        
        Returns:
            'sol', 'sag', 'arka', veya None (belirlenemezse)
        """
        if not start_room_center or len(path_points) < 3:
            return None
        
        # ── Koridor junction ve kapı bilgisini bul ──
        corridor_junction = None
        corridor_next = None
        door_room_side_point = None  # Kapının oda tarafındaki ucu
        
        if path_conn_ids and graph:
            conn_lookup = {c['id']: c for c in graph.connections}
            last_door_conn = None
            
            for i, conn_id in enumerate(path_conn_ids):
                conn = conn_lookup.get(conn_id)
                if not conn:
                    continue
                    
                if conn['type'] == 'door':
                    last_door_conn = conn
                    continue
                
                # İlk koridor (non-door) connection bulundu
                p1 = (conn['x1'], conn['y1'])
                p2 = (conn['x2'], conn['y2'])
                
                # Hangi uç önceki connection'a (kapıya) yakın?
                if i > 0:
                    prev_conn = conn_lookup.get(path_conn_ids[i - 1])
                    if prev_conn:
                        prev_ends = [(prev_conn['x1'], prev_conn['y1']),
                                     (prev_conn['x2'], prev_conn['y2'])]
                        d_p1 = min(distance(p1, pe) for pe in prev_ends)
                        d_p2 = min(distance(p2, pe) for pe in prev_ends)
                        
                        if d_p1 < d_p2:
                            corridor_junction = p1
                            corridor_next = p2
                        else:
                            corridor_junction = p2
                            corridor_next = p1
                    else:
                        corridor_junction = p1
                        corridor_next = p2
                else:
                    corridor_junction = p1
                    corridor_next = p2
                
                # Kapının oda tarafındaki ucunu bul
                # Kapı'nın iki ucundan, koridor junction'a UZAK olan ucu
                # odanın içindedir — o noktayı kullan
                if last_door_conn and corridor_junction:
                    dp1 = (last_door_conn['x1'], last_door_conn['y1'])
                    dp2 = (last_door_conn['x2'], last_door_conn['y2'])
                    d1_to_corr = distance(dp1, corridor_junction)
                    d2_to_corr = distance(dp2, corridor_junction)
                    # Koridor junction'a uzak olan uç = oda tarafı
                    door_room_side_point = dp1 if d1_to_corr > d2_to_corr else dp2
                
                break
        
        # ── Koridor yönünü belirle ──
        if corridor_junction and corridor_next:
            ref_point = corridor_junction
            
            # path_points üzerinde corridor_junction'a en yakın noktayı
            # bul ve oradan en az 50px ileriye bak
            MIN_WALK_DIST = 50.0
            
            best_idx = 0
            best_dist = float('inf')
            for k, pp in enumerate(path_points):
                d = distance(pp, corridor_junction)
                if d < best_dist:
                    best_dist = d
                    best_idx = k
            
            walk_ref_idx = min(best_idx + 1, len(path_points) - 1)
            cumulative = 0.0
            for k in range(best_idx, len(path_points) - 1):
                seg_len = math.hypot(
                    path_points[k + 1][0] - path_points[k][0],
                    path_points[k + 1][1] - path_points[k][1]
                )
                cumulative += seg_len
                if cumulative >= MIN_WALK_DIST:
                    walk_ref_idx = k + 1
                    break
            else:
                walk_ref_idx = len(path_points) - 1
            
            walk_dx = path_points[walk_ref_idx][0] - ref_point[0]
            walk_dy = path_points[walk_ref_idx][1] - ref_point[1]
        else:
            # Fallback: path_points[1]'den itibaren
            ref_point = path_points[1]
            MIN_WALK_DIST = 50.0
            walk_ref_idx = 2
            cumulative = 0.0
            for k in range(1, len(path_points) - 1):
                seg_len = math.hypot(
                    path_points[k + 1][0] - path_points[k][0],
                    path_points[k + 1][1] - path_points[k][1]
                )
                cumulative += seg_len
                if cumulative >= MIN_WALK_DIST:
                    walk_ref_idx = k + 1
                    break
            else:
                walk_ref_idx = len(path_points) - 1
            
            walk_dx = path_points[walk_ref_idx][0] - ref_point[0]
            walk_dy = path_points[walk_ref_idx][1] - ref_point[1]
        
        walk_mag = math.hypot(walk_dx, walk_dy)
        if walk_mag < 1.0:
            return None
        
        walk_dx /= walk_mag
        walk_dy /= walk_mag
        
        # ── Mağaza pozisyonunu hesapla ──
        # Kapının oda tarafındaki ucunu kullan (varsa).
        # Bu nokta koridorun doğru tarafında olduğu garanti —
        # büyük polygon merkezlerinden çok daha güvenilir.
        room_indicator = door_room_side_point if door_room_side_point else start_room_center
        
        store_dx = room_indicator[0] - ref_point[0]
        store_dy = room_indicator[1] - ref_point[1]
        
        store_mag = math.hypot(store_dx, store_dy)
        if store_mag < 1.0:
            return None
        
        # ── Çapraz çarpım: koridorun hangi tarafında? ──
        cross = walk_dx * store_dy - walk_dy * store_dx
        
        # Açı tabanlı kontrol: sin_angle küçükse mağaza koridorun
        # doğrusal devamında → 'arka'
        sin_angle = abs(cross) / store_mag
        
        if sin_angle < 0.18:
            return 'arka'
        
        return 'sol' if cross < 0 else 'sag'
    
    def _calculate_total_distance(self, path_points: List[Tuple[float, float]]) -> float:
        """Toplam rota mesafesini hesapla"""
        total = 0
        for i in range(len(path_points) - 1):
            total += distance(path_points[i], path_points[i + 1]) * self.pixel_to_meter_ratio
        return total
    
    def _calculate_segment_distance(self, path_points: List[Tuple[float, float]], 
                                    start_idx: int, end_idx: int) -> float:
        """İki index arasındaki mesafeyi hesapla"""
        if start_idx is None or end_idx is None or start_idx >= end_idx:
            return 0
        
        segment_dist = 0
        for i in range(start_idx, min(end_idx, len(path_points) - 1)):
            segment_dist += distance(path_points[i], path_points[i + 1]) * self.pixel_to_meter_ratio
        return segment_dist
    
    def _find_turn_index_in_path(self, path_points: List[Tuple[float, float]], 
                                  turn_point: Tuple[float, float], 
                                  epsilon: float = 1.0) -> Optional[int]:
        """Dönüş noktasının path_points içindeki index'ini bul"""
        for i, point in enumerate(path_points):
            if distance(point, turn_point) < epsilon:
                return i
        return None
    
    def _find_first_turn_index(self, path_points: List[Tuple[float, float]], 
                               turns: List[Dict]) -> int:
        """İlk dönüş noktasının index'ini bul"""
        if not turns:
            return len(path_points) - 1
        
        first_turn_index = self._find_turn_index_in_path(path_points, turns[0]['point'])
        return first_turn_index if first_turn_index is not None else len(path_points) - 1
    
    def _calculate_bearing(self, point1: Tuple[float, float], 
                          point2: Tuple[float, float]) -> float:
        """İki nokta arasındaki yönü (bearing) hesapla (derece olarak)"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        angle = math.atan2(dy, dx) * (180 / math.pi)
        # SVG koordinat sisteminde Y ekseni aşağı doğru, bunu düzelt
        bearing = (90 - angle) % 360
        return bearing
    
    def _bearing_to_direction(self, bearing: float) -> str:
        """Bearing'i yön açıklamasına çevir"""
        directions = [
            (22.5, "kuzey"),
            (67.5, "kuzeydoğu"),
            (112.5, "doğu"),
            (157.5, "güneydoğu"),
            (202.5, "güney"),
            (247.5, "güneybatı"),
            (292.5, "batı"),
            (337.5, "kuzeybatı"),
            (360, "kuzey")
        ]
        
        for angle, direction in directions:
            if bearing < angle:
                return direction
        return "kuzey"
    
    def _create_veer_description(self, direction: str,
                                 landmark: Optional[str],
                                 anchor_side: Optional[str],
                                 distance_after: float) -> str:
        """Zigzag alanı için hafif yönelme açıklaması oluştur"""
        import random
        
        # Yön metni
        if direction == 'düz':
            veer_text = "Düz"
        else:
            side = "sola" if direction == "sola" else "sağa"
            veer_phrases = [
                f"Hafif {side} yönelerek",
                f"Hafif {side} kayarak",
            ]
            veer_text = random.choice(veer_phrases)
        
        # Landmark eklentisi
        if landmark and anchor_side:
            side_text = "solunuzda" if anchor_side == "sol" else "sağınızda"
            veer_text += f" ({landmark} {side_text})"
        elif landmark:
            veer_text += f" ({landmark} yanından geçerek)"
        
        # Mesafe
        if distance_after > 0:
            movement = random.choice(["ilerleyin", "devam edin", "yürüyün"])
            if distance_after < 10:
                veer_text += f" {distance_after:.1f} metre {movement}"
            elif distance_after < 50:
                veer_text += f" yaklaşık {int(distance_after)} metre {movement}"
            else:
                veer_text += f" {int(distance_after)} metre {movement}"
        else:
            veer_text += " ilerleyin"
        
        return veer_text
    
    def _create_pass_by_description(self, landmark: Optional[str],
                                    anchor_side: Optional[str],
                                    distance_after: float) -> str:
        """Koridor kırılımı için landmark referanslı devam açıklaması oluştur"""
        import random
        
        if landmark and anchor_side:
            side_text = "solunuzda" if anchor_side == "sol" else "sağınızda"
            phrases = [
                f"{landmark} {side_text} kalacak şekilde",
                f"{landmark} {side_text} bırakarak",
                f"{landmark} {side_text} göreceksiniz,",
            ]
            base = random.choice(phrases)
        elif landmark:
            phrases = [
                f"{landmark} yanından geçerek",
                f"{landmark} önünden geçerek",
            ]
            base = random.choice(phrases)
        else:
            base = "Düz"
        
        if distance_after > 0:
            movement = random.choice(["ilerleyin", "devam edin", "yürüyün"])
            if distance_after < 10:
                base += f" {distance_after:.1f} metre {movement}"
            elif distance_after < 50:
                base += f" yaklaşık {int(distance_after)} metre {movement}"
            else:
                base += f" {int(distance_after)} metre {movement}"
        else:
            base += " ilerleyin"
        
        return base
    
    def _create_turn_description(self, turn_direction: str, 
                                 distance_after: float, 
                                 landmark: Optional[str] = None) -> str:
        """Metrik dönüş açıklaması oluştur - sadece mesafe odaklı"""
        import random
        
        # Dönüş ifadeleri - çeşitlendirilmiş
        turn_phrases = {
            "sağa": [
                "Sağa dönün",
                "Sağ tarafa dönün",
                "Sağa",
                "Sağ tarafa sapın",
                "Sağdan gidin",
                "Sağa yönelin",
                "Sağ yöne dönün"
            ],
            "sola": [
                "Sola dönün",
                "Sol tarafa dönün",
                "Sola",
                "Sol tarafa sapın",
                "Soldan gidin",
                "Sola yönelin",
                "Sol yöne dönün"
            ]
        }
        
        base_text = random.choice(turn_phrases.get(turn_direction, [f"{turn_direction.capitalize()} dönün"]))
        
        if distance_after > 0:
            # Bağlaç çeşitlendirmesi
            connectors = [" ve ", ", ", " sonra ", " ardından ", ", daha sonra "]
            connector = random.choice(connectors)
            
            # Hareket fiilleri çeşitlendirmesi
            movement_verbs = [
                "ilerleyin",
                "devam edin",
                "gidin",
                "yürüyün",
                "düz gidin",
                "düz ilerleyin",
                "ilerlemeye devam edin",
                "yolunuza devam edin"
            ]
            
            # Mesafe ifadeleri - çeşitlendirilmiş
            if distance_after < 10:
                # Kısa mesafeler için hassas ifadeler
                distance_expressions = [
                    f"{distance_after:.1f} metre {random.choice(movement_verbs)}",
                    f"tam {distance_after:.1f} metre {random.choice(movement_verbs)}",
                    f"{distance_after:.1f} m {random.choice(movement_verbs)}",
                    f"{distance_after:.1f} metre boyunca {random.choice(movement_verbs[:4])}"
                ]
                base_text += connector + random.choice(distance_expressions)
                
            elif distance_after < 50:
                # Orta mesafeler için yaklaşık ifadeler
                distance_expressions = [
                    f"yaklaşık {int(distance_after)} metre {random.choice(movement_verbs)}",
                    f"tahminen {int(distance_after)} m {random.choice(movement_verbs)}",
                    f"kabaca {int(distance_after)} metre {random.choice(movement_verbs)}",
                    f"yaklaşık {int(distance_after)} metre boyunca {random.choice(movement_verbs[:4])}",
                    f"{int(distance_after)} metre kadar {random.choice(movement_verbs)}"
                ]
                base_text += connector + random.choice(distance_expressions)
                
            else:
                # Uzun mesafeler için daha genel ifadeler
                distance_expressions = [
                    f"{int(distance_after)} metre düz gidin",
                    f"{int(distance_after)} m {random.choice(movement_verbs)}",
                    f"yaklaşık {int(distance_after)} metre {random.choice(movement_verbs)}",
                    f"{int(distance_after)} metre boyunca düz {random.choice(movement_verbs[:4])}",
                    f"{int(distance_after)} metre ileriye doğru {random.choice(movement_verbs[:4])}"
                ]
                base_text += connector + random.choice(distance_expressions)
        
        return base_text
    
    def print_directions(self):
        """Metrik yol tarifini konsola yazdır"""
        print("\n" + "="*70)
        print("METRİK TABANLI ROTA TARİFİ (GPS Benzeri)")
        print("="*70)
        
        if not self.steps:
            print("Henüz rota tarifleri oluşturulmadı.")
            return
        
        total_distance = self.steps[-1].cumulative_distance
        print(f"Toplam Mesafe: {total_distance:.1f} metre")
        print(f"Toplam Adım Sayısı: {len(self.steps)}")
        print("="*70 + "\n")
        
        for step in self.steps:
            print(f"{step}")
            
            # Detay bilgilerini göster
            if step.distance_meters > 0:
                print(f"   Mesafe: {step.distance_meters:.1f}m")
            
            if step.landmark:
                print(f"   Referans: {step.landmark}")
            
            if step.cumulative_distance > 0:
                print(f"   Kalan: {(total_distance - step.cumulative_distance):.1f}m")
            
            print()
        
        print("="*70)
    
    def export_to_json(self) -> List[Dict]:
        """Tarifleri JSON formatında dışa aktar"""
        return [step.to_dict() for step in self.steps]
    
    def get_summary(self) -> Dict:
        """Rota özeti döndür"""
        if not self.steps:
            return {}
        
        return {
            "total_distance_meters": round(self.steps[-1].cumulative_distance, 1),
            "total_steps": len(self.steps),
            "estimated_time_minutes": round(self.steps[-1].cumulative_distance / 80, 1),  # Ortalama yürüme hızı ~80m/dk
            "turns_count": sum(1 for step in self.steps if "TURN" in step.action),
            "landmarks_count": sum(1 for step in self.steps if step.landmark is not None)
        }
