"""
Multi-Floor Route Generator - Katlar arası rota üretimi
"""
import math
from typing import List, Dict, Tuple, Optional
from helpers.dijkstra import dijkstra_connections
from helpers.path_analysis import distance, NavigationGraphCleaner
from helpers.route_directions import MetricRouteGenerator
from helpers.alternative_routes import dijkstra_with_custom_cost
from helpers.portal_matcher import (
    parse_portal_name, find_matching_portal, plan_floor_transitions,
    find_nearest_portal_to_point, find_available_portals_from_floor,
    get_floor_number
)
from helpers.batch_route_generator import (
    get_connection_points, is_significant_turn, format_angle_direction,
    find_nearest_largest_room, _detect_turns_from_path
)


def generate_multi_floor_route(start_room: Dict,
                               end_room: Dict,
                               start_graph,
                               end_graph,
                               all_graphs: List,
                               floor_names: List[str],
                               floor_areas_list: List[Dict],
                               start_floor_name: str,
                               end_floor_name: str,
                               pixel_to_meter_ratio: float = 0.1) -> Optional[Dict]:
    """
    Katlar arası rota oluşturur
    
    Args:
        start_room: Başlangıç odası
        end_room: Hedef odası
        start_graph: Başlangıç katı graph'ı
        end_graph: Hedef katı graph'ı
        all_graphs: Tüm katların graph'ları
        floor_names: Kat isimleri
        floor_areas_list: Her katın oda bilgileri
        start_floor_name: Başlangıç kat ismi
        end_floor_name: Hedef kat ismi
        pixel_to_meter_ratio: Piksel-metre oranı
    
    Returns:
        Rota bilgileri dictionary
    """
    from helpers.batch_route_generator import find_nearest_connection_to_room
    
    # Başlangıç noktası (en yakın portalı seçmek için)
    start_point = start_room.get('center') if start_room else None
    
    # 1. Kat geçiş planı oluştur (başlangıç noktasına en yakın portalı seçecek)
    floor_transitions = plan_floor_transitions(
        start_floor_name, end_floor_name, all_graphs, floor_names, start_point
    )
    
    # Aynı kattaysak normal rota
    if not floor_transitions:
        print("  Aynı kat içi rota")
        return None
    
    print(f"  Kat geçişleri: {len(floor_transitions)} adım")
    
    # 2. Her kat için segment'leri oluştur
    all_segments = []
    current_room = start_room
    current_graph = start_graph
    current_floor = start_floor_name
    current_floor_areas = floor_areas_list[floor_names.index(start_floor_name)]
    
    # İlk segmenti ekle: Başlangıç -> İlk Portal
    first_transition = floor_transitions[0]
    from_floor, to_floor, portal_data = first_transition
    
    # İlk portala en yakın connection'ı bul
    first_portal_conn = portal_data['connection']
    first_portal_point = ((first_portal_conn['x1'] + first_portal_conn['x2']) / 2,
                          (first_portal_conn['y1'] + first_portal_conn['y2']) / 2)
    
    # Başlangıç noktası (önce odanın kendi kapısını ara)
    start_conn_id = find_nearest_connection_to_room(current_room['center'], current_graph, room_id=current_room['id'])
    
    # İlk portala rota
    segment = create_route_segment(
        start_conn_id=start_conn_id,
        end_conn_id=first_portal_conn['id'],
        graph=current_graph,
        floor_areas=current_floor_areas,
        start_desc=f"{current_room['type']} - {current_room['id']}",
        end_desc=f"Portal: {portal_data['portal_info']['type']} {portal_data['portal_info']['number']}",
        floor_name=current_floor,
        pixel_to_meter_ratio=pixel_to_meter_ratio,
        start_room_center=start_room.get('center')
    )
    
    if segment:
        all_segments.append(segment)
    
    # 3. Portal geçişlerini işle
    for i, (from_floor, to_floor, portal_data) in enumerate(floor_transitions):
        # Hedef kattaki eşleşen portalı bul
        from_floor_idx = floor_names.index(from_floor)
        to_floor_idx = floor_names.index(to_floor)
        
        to_graph = all_graphs[to_floor_idx]
        to_floor_areas = floor_areas_list[to_floor_idx]
        
        # Eşleşen portalı bul
        portal_info = portal_data['portal_info']
        matching_portal = find_matching_portal(
            portal_info, 
            str(get_floor_number(from_floor)),
            to_graph.connections
        )
        
        if not matching_portal:
            print(f"  Uyarı: {to_floor} katında eşleşen portal bulunamadı")
            continue
        
        # Portal geçiş adımını ekle
        portal_transition_step = {
            'type': 'portal_transition',
            'from_floor': from_floor,
            'to_floor': to_floor,
            'portal_type': portal_info['type'],
            'portal_number': portal_info['number'],
            'description': f"{portal_info['type']} {portal_info['number']} ile {to_floor} katına geçin"
        }
        all_segments.append(portal_transition_step)
        
        # Son geçişse, hedef odaya git
        if i == len(floor_transitions) - 1:
            # Hedef odaya rota (önce odanın kendi kapısını ara)
            end_conn_id = find_nearest_connection_to_room(end_room['center'], to_graph, room_id=end_room['id'])
            
            segment = create_route_segment(
                start_conn_id=matching_portal['id'],
                end_conn_id=end_conn_id,
                graph=to_graph,
                floor_areas=to_floor_areas,
                start_desc=f"Portal: {portal_info['type']} {portal_info['number']}",
                end_desc=f"{end_room['type']} - {end_room['id']}",
                floor_name=to_floor,
                pixel_to_meter_ratio=pixel_to_meter_ratio
            )
            
            if segment:
                all_segments.append(segment)
        else:
            # Ara kata gidiyoruz, sonraki portala git
            next_transition = floor_transitions[i + 1]
            next_portal_data = next_transition[2]
            next_portal_conn = next_portal_data['connection']
            
            segment = create_route_segment(
                start_conn_id=matching_portal['id'],
                end_conn_id=next_portal_conn['id'],
                graph=to_graph,
                floor_areas=to_floor_areas,
                start_desc=f"Portal: {portal_info['type']} {portal_info['number']}",
                end_desc=f"Portal: {next_portal_data['portal_info']['type']} {next_portal_data['portal_info']['number']}",
                floor_name=to_floor,
                pixel_to_meter_ratio=pixel_to_meter_ratio
            )
            
            if segment:
                all_segments.append(segment)
    
    # 4. Tüm segment'leri birleştir
    return combine_segments(all_segments, start_room, end_room, floor_transitions)


def create_route_segment(start_conn_id: str,
                        end_conn_id: str,
                        graph,
                        floor_areas: Dict,
                        start_desc: str,
                        end_desc: str,
                        floor_name: str,
                        pixel_to_meter_ratio: float = 0.1,
                        route_type: str = "distance",
                        start_room_center: Optional[Tuple[float, float]] = None) -> Optional[Dict]:
    """
    İki connection arasında rota segment'i oluşturur
    
    Args:
        route_type: "distance" (en kısa) veya "turns" (en az dönüş)
        start_room_center: Başlangıç odasının merkez koordinatları (mağaza tarafı belirlemek için)
    """
    if not start_conn_id or not end_conn_id:
        return None
    
    # Aynı noktaysa atla
    if start_conn_id == end_conn_id:
        return None
    
    # Alternatif rota hesaplama ile yol bul
    try:
        result = dijkstra_with_custom_cost(graph, start_conn_id, end_conn_id, route_type)
        if result:
            path, metrics = result
        else:
            return None
    except Exception as e:
        print(f"    Segment rotası bulunamadı: {str(e)}")
        return None
    
    if not path or len(path) < 1:
        return None
    
    # Navigation graph cleaner oluştur (veya cache'den al)
    if not hasattr(graph, '_nav_cleaner'):
        graph._nav_cleaner = NavigationGraphCleaner(graph)
    
    # Chain-based temizlenmiş path_points oluştur
    path_points = graph._nav_cleaner.clean_path(path)
    
    if len(path_points) < 2:
        return None
    
    # Dönüş noktalarını bul
    all_turns = _detect_turns_from_path(path_points, path, graph, floor_areas)
    
    # Metrik tarif oluştur
    metric_generator = MetricRouteGenerator(pixel_to_meter_ratio=pixel_to_meter_ratio)
    metric_generator.generate_directions(
        path_points=path_points,
        turns=all_turns,
        start_location=start_desc,
        end_location=end_desc,
        start_room_center=start_room_center,
        path_conn_ids=path,
        graph=graph
    )
    
    summary = metric_generator.get_summary()
    
    return {
        'type': 'route_segment',
        'floor': floor_name,
        'from': start_desc,
        'to': end_desc,
        'summary': summary,
        'steps': metric_generator.export_to_json(),
        'path_connections': path,  # Connection ID'leri - görselleştirme için
        'path_points': path_points,
        'turns': all_turns
    }


def combine_segments(segments: List[Dict],
                    start_room: Dict,
                    end_room: Dict,
                    floor_transitions: List) -> Dict:
    """
    Tüm segment'leri birleştirerek tek bir rota oluşturur
    """
    # Toplam istatistikleri hesapla
    total_distance = 0
    total_turns = 0
    all_steps = []
    step_number = 1
    
    for segment in segments:
        if segment['type'] == 'portal_transition':
            # Portal geçişi adımı
            all_steps.append({
                'step_number': step_number,
                'action': 'FLOOR_CHANGE',
                'description': segment['description'],
                'from_floor': segment['from_floor'],
                'to_floor': segment['to_floor'],
                'portal_type': segment['portal_type']
            })
            step_number += 1
        else:
            # Normal rota segmenti
            total_distance += segment['summary']['total_distance_meters']
            total_turns += len(segment['turns'])
            
            # Adımları ekle
            for step in segment['steps']:
                step['step_number'] = step_number
                all_steps.append(step)
                step_number += 1

    # Portal varış/çıkış adımlarını yeniden adlandır
    for i, step in enumerate(all_steps):
        if (step.get('action') == 'ARRIVE'
                and i + 1 < len(all_steps)
                and all_steps[i + 1].get('action') == 'FLOOR_CHANGE'):
            step['action'] = 'ARRIVE_PORTAL'
        elif (step.get('action') == 'START'
              and i - 1 >= 0
              and all_steps[i - 1].get('action') == 'FLOOR_CHANGE'):
            step['action'] = 'START_PORTAL'
            step['portal_type'] = all_steps[i - 1].get('portal_type', '')

    # Tüm path_points ve turns'leri birleştir
    all_path_points = []
    all_turns = []
    
    for segment in segments:
        if segment['type'] == 'route_segment':
            # Segment'in path noktalarını ekle
            if 'path_points' in segment and segment['path_points']:
                # İlk segment değilse, son nokta ile yeni segmentin ilk noktası aynıysa biri atlanır
                if all_path_points and segment['path_points']:
                    # Eğer son nokta ile yeni ilk nokta çok yakınsa (aynı portal), birini atla
                    if len(all_path_points) > 0:
                        last_point = all_path_points[-1]
                        first_new_point = segment['path_points'][0]
                        from helpers.path_analysis import distance
                        if distance(last_point, first_new_point) < 5:  # Çok yakınsa
                            all_path_points.extend(segment['path_points'][1:])
                        else:
                            all_path_points.extend(segment['path_points'])
                    else:
                        all_path_points.extend(segment['path_points'])
                else:
                    all_path_points.extend(segment['path_points'])
            
            # Segment'in turn'lerini ekle
            if 'turns' in segment and segment['turns']:
                all_turns.extend(segment['turns'])
    
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
        'is_multi_floor': True,
        'floor_transitions': len(floor_transitions),
        'summary': {
            'total_distance_meters': total_distance,
            'total_steps': len(all_steps),
            'estimated_time_minutes': round(total_distance / 80 + len(floor_transitions) * 0.5, 1),  # Portal geçişi +30 saniye
            'turns_count': total_turns,
            'floor_changes': len(floor_transitions)
        },
        'steps': all_steps,
        'segments': segments,
        'path_points': all_path_points,  # Tüm path noktaları
        'turns': all_turns  # Tüm dönüş noktaları
    }




