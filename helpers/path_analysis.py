import math
from typing import List, Tuple, Dict, Set
import xml.etree.ElementTree as ET

def round_point(pt: Tuple[float, float]) -> Tuple[float, float]:
    """
    Rounds the coordinates of a point to 3 decimal places.
    """
    return (round(pt[0], 3), round(pt[1], 3))

def distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Calculates the Euclidean distance between two 2D points.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])

def calculate_area(points: List[Tuple[float, float]]) -> float:
    """
    Calculates the area of a polygon using the shoelace formula.
    """
    if len(points) < 3:
        return 0
    area = 0.0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0

def find_sharp_turns(path_points: List[Tuple[float, float]], angle_threshold: float = 89) -> Tuple[List[Tuple[Tuple[float, float], float]], List[Tuple[Tuple[float, float], float]]]:
    """
    Identifies sharp and gentle turns in a path based on angle between consecutive segments.
    Returns two lists of tuples: (turn_point, angle) for both sharp and gentle turns.
    """
    sharp_turns = []
    gentle_turns = []

    if len(path_points) < 3:
        return sharp_turns, gentle_turns

    for i in range(1, len(path_points) - 1):
        a = path_points[i - 1]  # Previous point
        b = path_points[i]      # Current turn point
        c = path_points[i + 1]  # Next point

        # Vectors representing the two segments
        v1 = (b[0] - a[0], b[1] - a[1])
        v2 = (c[0] - b[0], c[1] - b[1])

        # Magnitudes of the vectors
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)

        if mag1 == 0 or mag2 == 0:
            continue

        # Calculate angle using dot product
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_theta = dot / (mag1 * mag2)
        cos_theta = max(min(cos_theta, 1), -1)  # Clamp to [-1, 1]
        angle = math.acos(cos_theta) * (180 / math.pi)

        if angle >= angle_threshold:
            sharp_turns.append((b, angle))
        else:
            gentle_turns.append((b, angle))

    return sharp_turns, gentle_turns

def find_nearby_anchors(turn_points: List[Tuple[float, float]], 
                       anchor_points: Dict[str, Dict],
                       distance_threshold: float = 100,
                       min_distance_between_anchors: float = 200) -> List[Dict]:
    """
    Finds suitable anchor points near turn points with constraints.
    """
    all_matches = []
    
    for turn_point in turn_points:
        for anchor_id, anchor_data in anchor_points.items():
            min_dist = float('inf')
            
            # Calculate minimum distance between turn point and anchor points
            for pt in anchor_data["points"]:
                min_dist = min(min_dist, distance(pt, turn_point))
                
            if min_dist < distance_threshold:
                all_matches.append({
                    "id": anchor_id,
                    "area": anchor_data["area"],
                    "distance_to_turn": min_dist,
                    "centroid": anchor_data["centroid"],
                    "turn_point": turn_point
                })
    
    # Sort by area (descending) and distance (ascending)
    all_matches.sort(key=lambda x: (-x["area"], x["distance_to_turn"]))
    
    # Filter anchors that are too close to each other
    selected_anchors = []
    used_centroids = []
    
    for match in all_matches:
        is_too_close = False
        for used_centroid in used_centroids:
            if distance(match["centroid"], used_centroid) < min_distance_between_anchors:
                is_too_close = True
                break
                
        if not is_too_close:
            selected_anchors.append(match)
            used_centroids.append(match["centroid"])
            
    return selected_anchors

# =====================================================================
# CONNECTION PROBLEM SCORING
# =====================================================================

def compute_connection_problem_scores(graph) -> Dict[str, float]:
    """
    Her connection için sorun skoru hesaplar.
    Kısa segment + komşuyla büyük açı değişimi = yüksek skor.
    
    Bu skor, rota path_points oluşturulurken zigzag junction noktalarını
    atlamak için kullanılır.
    
    Returns:
        {connection_id: problem_score} dictionary
    """
    def _bearing(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(dx, -dy)) % 360
    
    def _angle_diff(b1, b2):
        diff = abs(b1 - b2)
        return min(diff, 360 - diff)
    
    # Connection bilgilerini hazırla
    conn_info = {}
    for conn in graph.connections:
        cid = conn['id']
        p1 = (conn['x1'], conn['y1'])
        p2 = (conn['x2'], conn['y2'])
        length = distance(p1, p2)
        conn_info[cid] = {'p1': p1, 'p2': p2, 'length': length}
    
    problem_scores = {}
    
    for conn_id, neighbors in graph.adjacency_list.items():
        if conn_id not in conn_info:
            continue
        ci = conn_info[conn_id]
        
        for neighbor_id in neighbors:
            if neighbor_id not in conn_info:
                continue
            ni = conn_info[neighbor_id]
            
            # Paylaşılan noktayı bul
            shared = None
            conn_other = None
            neigh_other = None
            
            for cp in [ci['p1'], ci['p2']]:
                for np_ in [ni['p1'], ni['p2']]:
                    if abs(cp[0] - np_[0]) < 1.0 and abs(cp[1] - np_[1]) < 1.0:
                        shared = cp
                        conn_other = ci['p2'] if cp == ci['p1'] else ci['p1']
                        neigh_other = ni['p2'] if np_ == ni['p1'] else ni['p1']
                        break
                if shared:
                    break
            
            if not shared:
                continue
            
            # Approach / exit bearing
            approach = _bearing(conn_other, shared)
            exit_b = _bearing(shared, neigh_other)
            turn = _angle_diff(approach, exit_b)
            
            # Kısa segment + büyük açı = sorunlu
            min_len = min(ci['length'], ni['length'])
            
            if min_len < 25 and turn > 30 and turn < 165:
                score = (turn / 10) * (30 / max(min_len, 0.5))
                
                for cid in [conn_id, neighbor_id]:
                    if cid not in problem_scores or problem_scores[cid] < score:
                        problem_scores[cid] = score
    
    return problem_scores


def build_smooth_path_points(path_conn_ids: List[str], 
                              graph,
                              problem_scores: Dict[str, float] = None,
                              problem_threshold: float = 15.0) -> List[Tuple[float, float]]:
    """
    Connection ID listesinden path_points oluşturur.
    Ardışık sorunlu connection'ların ara junction noktalarını atlayarak
    zigzag ve mikro-kırılımları ön-düzeltme yapar.
    
    Args:
        path_conn_ids: Rota connection ID'leri
        graph: Graph nesnesi
        problem_scores: compute_connection_problem_scores() çıktısı (None ise hesaplanır)
        problem_threshold: Bu skorun üzerindeki connection'lar sorunlu sayılır
    
    Returns:
        Düzleştirilmiş path noktaları listesi
    """
    if not path_conn_ids:
        return []
    
    # Problem skorları yoksa hesapla
    if problem_scores is None:
        problem_scores = compute_connection_problem_scores(graph)
    
    # Her connection'ın uç noktalarını al
    def get_points(conn_id):
        conn = next((c for c in graph.connections if c['id'] == conn_id), None)
        if conn:
            return (conn['x1'], conn['y1']), (conn['x2'], conn['y2'])
        return None, None
    
    # Önce ham path_points oluştur (klasik yöntem)
    raw_points = []
    conn_at_point = []  # Her point'in hangi connection junction'ında olduğunu izle
    
    for i, conn_id in enumerate(path_conn_ids):
        p1, p2 = get_points(conn_id)
        if not p1:
            continue
        
        if i == 0:
            raw_points.append(p1)
            conn_at_point.append((conn_id, 'start'))
            raw_points.append(p2)
            conn_at_point.append((conn_id, 'end'))
        else:
            # Önceki noktayla doğru ucu eşleştir
            prev = raw_points[-1]
            if distance(prev, p1) < distance(prev, p2):
                # p1 öncekine yakın → p2'yi ekle
                raw_points.append(p2)
                conn_at_point.append((conn_id, 'end'))
            else:
                # p2 öncekine yakın → p1'i ekle
                raw_points.append(p1)
                conn_at_point.append((conn_id, 'start'))
    
    if len(raw_points) < 2:
        return raw_points
    
    # Şimdi ön-düzeltme: ardışık sorunlu connection'ların ara noktalarını atla
    # Mantık: Eğer path_points[i] bir junction noktasıysa ve 
    # hem önceki hem sonraki connection sorunlu ise → bu noktayı atla
    
    smoothed = [raw_points[0]]  # Başlangıç noktası her zaman kalır
    
    for i in range(1, len(raw_points) - 1):
        # Bu noktanın önceki ve sonraki connection'ı
        # raw_points[i] → path_conn_ids[i-1] ile path_conn_ids[i] arasındaki junction
        if i - 1 < len(path_conn_ids) and i < len(path_conn_ids):
            prev_conn = path_conn_ids[i - 1]
            next_conn = path_conn_ids[i]
            
            prev_score = problem_scores.get(prev_conn, 0)
            next_score = problem_scores.get(next_conn, 0)
            
            # Her iki connection da sorunlu ise → junction noktasını atla
            if prev_score >= problem_threshold and next_score >= problem_threshold:
                continue
        
        smoothed.append(raw_points[i])
    
    smoothed.append(raw_points[-1])  # Bitiş noktası her zaman kalır
    
    return smoothed


# =====================================================================
# CHAIN-BASED NAVIGATION GRAPH CLEANER
# =====================================================================

class NavigationGraphCleaner:
    """
    Chain-based geometric analysis for cleaning navigation paths.
    
    SVG geometri verilerindeki zigzag, mikro-kırılım ve gürültüyü
    temizlemek için graph node derecelerine dayalı koridor zinciri analizi.
    
    Algoritma:
    1. Connection graph'ından node graph oluştur (endpoint merging)
    2. Her node'un derecesini hesapla
    3. Korunacak node'ları belirle (derece >= 3, kapı, portal, anchor)
    4. Ardışık derece-2 node'ları koridor zincirlerine grupla
    5. Her zincir için geometrik analiz:
       - Kümülatif dönüş açısı
       - Açı işaret tutarlılığı (zigzag tespiti)
       - Heading varyansı
    6. Kurallar:
       - Zigzag (küçük kümülatif, alternatif işaretler) → düzleştir
       - Smooth arc (büyük kümülatif, monotonik) → tek semantik dönüş
       - Gürültü (düşük kümülatif, yüksek varyans) → düzleştir
    """
    
    def __init__(self, graph, merge_tolerance: float = 1.0):
        """
        Args:
            graph: Graph nesnesi (connections, adjacency_list)
            merge_tolerance: Yakın noktaları birleştirme toleransı (piksel)
        """
        self.graph = graph
        self.merge_tolerance = merge_tolerance
        
        # Node graph yapıları
        self.nodes = {}            # node_id -> (x, y)
        self.node_degree = {}      # node_id -> int
        self.node_connections = {} # node_id -> set of conn_ids
        self.node_is_special = {} # node_id -> bool (kapı, portal, anchor)
        self.conn_nodes = {}       # conn_id -> (node_id_1, node_id_2)
        
        # Connection lookup cache
        self._conn_lookup = {}
        for conn in graph.connections:
            self._conn_lookup[conn['id']] = conn
        
        # Node graph'ını oluştur
        self._build_node_graph()
    
    def _find_or_create_node(self, x: float, y: float) -> int:
        """Tolerans dahilinde mevcut node'u bul veya yeni oluştur"""
        for nid, (nx, ny) in self.nodes.items():
            if abs(x - nx) < self.merge_tolerance and abs(y - ny) < self.merge_tolerance:
                return nid
        
        nid = len(self.nodes)
        self.nodes[nid] = (x, y)
        self.node_degree[nid] = 0
        self.node_connections[nid] = set()
        self.node_is_special[nid] = False
        return nid
    
    def _build_node_graph(self):
        """Tüm connection'lardan node graph oluştur"""
        for conn in self.graph.connections:
            cid = conn['id']
            p1 = (conn['x1'], conn['y1'])
            p2 = (conn['x2'], conn['y2'])
            
            n1 = self._find_or_create_node(*p1)
            n2 = self._find_or_create_node(*p2)
            
            self.conn_nodes[cid] = (n1, n2)
            self.node_connections[n1].add(cid)
            self.node_connections[n2].add(cid)
            
            # Özel node'ları işaretle (kapı, portal)
            if conn['type'] in ('door', 'portal'):
                self.node_is_special[n1] = True
                self.node_is_special[n2] = True
        
        # Dereceleri hesapla
        for nid in self.nodes:
            self.node_degree[nid] = len(self.node_connections[nid])
    
    def _is_preserved_node(self, node_id: int) -> bool:
        """
        Node'un korunması gerekip gerekmediğini kontrol et.
        
        Korunan node'lar:
        - Gerçek kavşaklar: 3+ path-tipi connection buluşuyor
        - Portal node'ları: asansör, merdiven geçiş noktaları
        
        Korunmayan (zincire dahil):
        - Koridor devamı: 2 path + N door (kapılar koridor kararını değiştirmez)
        """
        conns = self.node_connections.get(node_id, set())
        
        path_degree = 0
        has_portal = False
        
        for cid in conns:
            conn = self._conn_lookup.get(cid)
            if not conn:
                continue
            if conn['type'] == 'path':
                path_degree += 1
            elif conn['type'] == 'portal':
                has_portal = True
        
        # 3+ path connection = gerçek kavşak (karar noktası)
        if path_degree >= 3:
            return True
        
        # Portal = her zaman önemli
        if has_portal:
            return True
        
        return False
    
    def _bearing(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """p1'den p2'ye yön (derece, 0=Kuzey/Yukarı, saat yönünde)"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.degrees(math.atan2(dx, -dy)) % 360
    
    def _signed_angle_diff(self, b1: float, b2: float) -> float:
        """İşaretli açı farkı (b2 - b1), [-180, 180] aralığında"""
        diff = b2 - b1
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff
    
    def _build_path_node_sequence(self, path_conn_ids: List[str]) -> List[int]:
        """
        Path connection ID'lerinden sıralı node dizisi oluştur.
        Her connection iki node arasında bir kenar → n+1 node oluşur.
        """
        if not path_conn_ids:
            return []
        
        # İlk connection
        first_cid = path_conn_ids[0]
        if first_cid not in self.conn_nodes:
            return []
        
        n1, n2 = self.conn_nodes[first_cid]
        
        if len(path_conn_ids) == 1:
            return [n1, n2]
        
        # İkinci connection ile ilk connection'ın yönünü belirle
        second_cid = path_conn_ids[1]
        if second_cid not in self.conn_nodes:
            return [n1, n2]
        
        sn1, sn2 = self.conn_nodes[second_cid]
        
        # n2, ikinci connection'ın bir ucuyla eşleşiyor mu?
        if n2 == sn1 or n2 == sn2:
            nodes = [n1, n2]
        elif n1 == sn1 or n1 == sn2:
            nodes = [n2, n1]
        else:
            # Eşleşme yok - mesafeye göre belirle
            d_n2_sn1 = distance(self.nodes[n2], self.nodes[sn1])
            d_n2_sn2 = distance(self.nodes[n2], self.nodes[sn2])
            d_n1_sn1 = distance(self.nodes[n1], self.nodes[sn1])
            d_n1_sn2 = distance(self.nodes[n1], self.nodes[sn2])
            
            if min(d_n2_sn1, d_n2_sn2) < min(d_n1_sn1, d_n1_sn2):
                nodes = [n1, n2]
            else:
                nodes = [n2, n1]
        
        # Sonraki connection'ları işle
        for i in range(1, len(path_conn_ids)):
            cid = path_conn_ids[i]
            if cid not in self.conn_nodes:
                continue
            
            cn1, cn2 = self.conn_nodes[cid]
            last_node = nodes[-1]
            
            if last_node == cn1:
                nodes.append(cn2)
            elif last_node == cn2:
                nodes.append(cn1)
            else:
                # Gap - en yakın ucu paylaşılan say
                d1 = distance(self.nodes[last_node], self.nodes[cn1])
                d2 = distance(self.nodes[last_node], self.nodes[cn2])
                if d1 < d2:
                    nodes.append(cn2)
                else:
                    nodes.append(cn1)
        
        return nodes
    
    def _find_chains(self, path_nodes: List[int]) -> List[Dict]:
        """
        Path node dizisinde koridor zincirlerini bul.
        Zincir: İki korunan node arasındaki ardışık derece-2 node'lar.
        
        Returns:
            [{'start_node': int, 'end_node': int, 'interior_nodes': [int, ...]}]
        """
        if len(path_nodes) < 2:
            return []
        
        chains = []
        chain_start = 0  # İlk node her zaman zincir başlangıcı
        
        for i in range(1, len(path_nodes)):
            node = path_nodes[i]
            is_end = (i == len(path_nodes) - 1)
            is_preserved = self._is_preserved_node(node)
            
            if is_preserved or is_end:
                # Zinciri kapat
                interior = path_nodes[chain_start + 1 : i]
                chains.append({
                    'start_node': path_nodes[chain_start],
                    'end_node': node,
                    'interior_nodes': interior,
                    'start_idx': chain_start,
                    'end_idx': i
                })
                chain_start = i
        
        return chains
    
    def _analyze_chain(self, chain: Dict) -> Dict:
        """
        Bir koridor zincirini geometrik olarak analiz et.
        
        Returns:
            {
                'points': [(x,y), ...],     # Zincirdeki tüm noktalar
                'bearings': [float, ...],     # Her segment'in yönü
                'turning_angles': [float, ...], # Her iç düğümdeki dönüş açısı
                'cumulative_angle': float,    # Toplam kümülatif açı
                'max_individual': float,      # En büyük tekil açı
                'is_alternating': bool,       # Zigzag mı?
                'is_monotonic': bool,         # Tek yönlü mü?
                'heading_variance': float,    # Yön varyansı
                'chain_type': str             # 'zigzag', 'arc', 'noise', 'sharp', 'keep'
            }
        """
        all_node_ids = [chain['start_node']] + chain['interior_nodes'] + [chain['end_node']]
        points = [self.nodes[nid] for nid in all_node_ids]
        
        result = {
            'points': points,
            'bearings': [],
            'turning_angles': [],
            'cumulative_angle': 0,
            'max_individual': 0,
            'is_alternating': False,
            'is_monotonic': True,
            'heading_variance': 0,
            'chain_type': 'keep'
        }
        
        if len(points) < 2:
            return result
        
        # Segment bearing'lerini hesapla
        bearings = []
        for i in range(len(points) - 1):
            b = self._bearing(points[i], points[i + 1])
            bearings.append(b)
        result['bearings'] = bearings
        
        if len(bearings) < 2:
            return result
        
        # İç düğümlerdeki dönüş açılarını hesapla
        turning_angles = []
        for i in range(len(bearings) - 1):
            angle = self._signed_angle_diff(bearings[i], bearings[i + 1])
            turning_angles.append(angle)
        result['turning_angles'] = turning_angles
        
        if not turning_angles:
            return result
        
        # Metrikler
        cumulative = sum(turning_angles)
        abs_cumulative = abs(cumulative)
        max_individual = max(abs(a) for a in turning_angles)
        
        result['cumulative_angle'] = cumulative
        result['max_individual'] = max_individual
        
        # İşaret analizi
        pos_count = sum(1 for a in turning_angles if a > 5)
        neg_count = sum(1 for a in turning_angles if a < -5)
        neutral_count = len(turning_angles) - pos_count - neg_count
        
        is_alternating = (pos_count > 0 and neg_count > 0 and 
                         min(pos_count, neg_count) / max(pos_count, neg_count, 1) > 0.3)
        is_monotonic = (pos_count == 0 or neg_count == 0)
        
        result['is_alternating'] = is_alternating
        result['is_monotonic'] = is_monotonic
        
        # Heading varyansı (dairesel)
        if len(bearings) > 1:
            # Sinüs/kosinüs ortalaması ile dairesel varyans
            sin_sum = sum(math.sin(math.radians(b)) for b in bearings)
            cos_sum = sum(math.cos(math.radians(b)) for b in bearings)
            n = len(bearings)
            r = math.hypot(sin_sum / n, cos_sum / n)
            heading_variance = 1 - r  # 0 = tamamen tutarlı, 1 = tamamen dağınık
            result['heading_variance'] = heading_variance
        
        # Zincir tipini belirle
        # KURAL 0: Herhangi bir tekil açı çok keskinse → koruma modu
        if max_individual >= 45:
            result['chain_type'] = 'sharp'
            return result
        
        # KURAL 1: Zigzag tespiti
        # Küçük kümülatif açı + alternatif işaretler → zigzag
        if abs_cumulative < 20 and is_alternating:
            result['chain_type'] = 'zigzag'
            return result
        
        # KURAL 2: Smooth arc tespiti
        # Büyük kümülatif açı + monotonik yön değişimi
        if abs_cumulative >= 30 and is_monotonic:
            result['chain_type'] = 'arc'
            return result
        
        # KURAL 3: Gürültü tespiti
        # Düşük kümülatif açı + yüksek heading varyansı + salınımlı
        if abs_cumulative < 20 and heading_variance > 0.3:
            result['chain_type'] = 'noise'
            return result
        
        # KURAL 4: Varsayılan - önemli açıları koru
        result['chain_type'] = 'keep'
        return result
    
    def _simplify_chain(self, chain: Dict, analysis: Dict) -> List[Tuple[float, float]]:
        """
        Analiz sonucuna göre zinciri sadeleştir.
        
        Returns:
            Zincirin bitiş noktasına kadar eklenmesi gereken noktalar.
            (başlangıç noktası hariç, çünkü zaten önceki zincirden eklendi)
        """
        points = analysis['points']
        chain_type = analysis['chain_type']
        
        if len(points) < 2:
            return [points[-1]] if points else []
        
        if chain_type == 'zigzag':
            # Zigzag → düz çizgi (sadece bitiş noktası)
            return [points[-1]]
        
        if chain_type == 'noise':
            # Gürültü → düzleştir (sadece bitiş noktası)
            return [points[-1]]
        
        if chain_type == 'arc':
            # Smooth arc → maksimum eğrilik noktasında tek dönüş
            turning_angles = analysis['turning_angles']
            if turning_angles:
                max_idx = max(range(len(turning_angles)), 
                            key=lambda i: abs(turning_angles[i]))
                arc_point = points[max_idx + 1]  # +1 çünkü açı iki segment arasında
                return [arc_point, points[-1]]
            return [points[-1]]
        
        if chain_type == 'sharp':
            # Keskin dönüş(ler) var → önemli noktaları koru
            turning_angles = analysis['turning_angles']
            result = []
            for i, angle in enumerate(turning_angles):
                if abs(angle) >= 25:
                    result.append(points[i + 1])
            result.append(points[-1])
            # Tekrar eden son noktayı kaldır
            if len(result) >= 2 and distance(result[-1], result[-2]) < self.merge_tolerance:
                result = result[:-1]
            return result
        
        # 'keep' → varsayılan olarak kayda değer açı değişimlerini koru
        turning_angles = analysis['turning_angles']
        result = []
        for i, angle in enumerate(turning_angles):
            if abs(angle) >= 20:
                result.append(points[i + 1])
        result.append(points[-1])
        # Tekrar eden son noktayı kaldır
        if len(result) >= 2 and distance(result[-1], result[-2]) < self.merge_tolerance:
            result = result[:-1]
        return result
    
    def clean_path(self, path_conn_ids: List[str]) -> List[Tuple[float, float]]:
        """
        Path connection ID'lerinden temizlenmiş path noktaları üret.
        
        Süreç:
        1. Path node dizisini oluştur
        2. Koridor zincirlerini bul
        3. Her zinciri analiz et ve sadeleştir
        4. Temiz path_points listesi döndür
        
        Args:
            path_conn_ids: Rota connection ID'leri
        
        Returns:
            Temizlenmiş path noktaları listesi [(x, y), ...]
        """
        if not path_conn_ids:
            return []
        
        # 1. Node dizisini oluştur
        path_nodes = self._build_path_node_sequence(path_conn_ids)
        
        if len(path_nodes) < 2:
            # Fallback: ham noktaları döndür
            return self._fallback_path_points(path_conn_ids)
        
        # 2. Koridor zincirlerini bul
        chains = self._find_chains(path_nodes)
        
        if not chains:
            return self._fallback_path_points(path_conn_ids)
        
        # 3. Her zinciri analiz et ve sadeleştir
        cleaned_points = [self.nodes[path_nodes[0]]]  # Başlangıç noktası
        
        for chain in chains:
            analysis = self._analyze_chain(chain)
            simplified = self._simplify_chain(chain, analysis)
            
            for pt in simplified:
                # Tekrarlanan noktaları atla
                if cleaned_points and distance(cleaned_points[-1], pt) < self.merge_tolerance:
                    continue
                cleaned_points.append(pt)
        
        # Son nokta her zaman dahil olmalı
        end_point = self.nodes[path_nodes[-1]]
        if cleaned_points and distance(cleaned_points[-1], end_point) >= self.merge_tolerance:
            cleaned_points.append(end_point)
        
        return cleaned_points
    
    def _fallback_path_points(self, path_conn_ids: List[str]) -> List[Tuple[float, float]]:
        """Fallback: connection uç noktalarından basit path_points oluştur"""
        points = []
        for i, cid in enumerate(path_conn_ids):
            conn = self._conn_lookup.get(cid)
            if not conn:
                continue
            
            p1 = (conn['x1'], conn['y1'])
            p2 = (conn['x2'], conn['y2'])
            
            if i == 0:
                points.append(p1)
                points.append(p2)
            else:
                prev = points[-1]
                if distance(prev, p1) < distance(prev, p2):
                    points.append(p2)
                else:
                    points.append(p1)
        
        return points
    
    def get_node_info(self, path_conn_ids: List[str]) -> Dict:
        """
        Debug: Path node'larının bilgilerini döndür
        """
        path_nodes = self._build_path_node_sequence(path_conn_ids)
        chains = self._find_chains(path_nodes)
        
        info = {
            'total_nodes': len(path_nodes),
            'preserved_nodes': sum(1 for n in path_nodes if self._is_preserved_node(n)),
            'chains': []
        }
        
        for chain in chains:
            analysis = self._analyze_chain(chain)
            info['chains'].append({
                'start': self.nodes[chain['start_node']],
                'end': self.nodes[chain['end_node']],
                'interior_count': len(chain['interior_nodes']),
                'type': analysis['chain_type'],
                'cumulative_angle': round(analysis['cumulative_angle'], 1),
                'max_individual': round(analysis['max_individual'], 1),
                'heading_variance': round(analysis['heading_variance'], 3)
            })
        
        return info


def extract_path_points(path_element: ET.Element, namespace: Dict[str, str]) -> List[Tuple[float, float]]:
    """
    Extracts points from an SVG path element.
    """
    points = []
    d_attr = path_element.get('d')
    if not d_attr:
        return points
        
    # Basic path command parsing (M, L commands only)
    commands = d_attr.split()
    current_point = None
    i = 0
    
    while i < len(commands):
        cmd = commands[i]
        if cmd in ['M', 'm', 'L', 'l']:
            try:
                x = float(commands[i + 1])
                y = float(commands[i + 2])
                
                if cmd.islower():  # Relative coordinates
                    if current_point:
                        x += current_point[0]
                        y += current_point[1]
                
                current_point = (x, y)
                points.append(current_point)
                i += 3
            except (IndexError, ValueError):
                i += 1
        else:
            i += 1
            
    return points 