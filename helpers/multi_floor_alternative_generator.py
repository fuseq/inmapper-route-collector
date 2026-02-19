"""
Multi-Floor Alternative Route Generator - Çok katlı rotalar için alternatif hesaplar
"""
from typing import Dict, List, Optional
from helpers.multi_floor_route_generator import (
    generate_multi_floor_route, create_route_segment
)
from helpers.batch_route_generator import find_nearest_connection_to_room
from helpers.portal_matcher import plan_floor_transitions, find_matching_portal, get_floor_number


def generate_multi_floor_alternatives(start_room: Dict,
                                      end_room: Dict,
                                      start_graph,
                                      end_graph,
                                      all_graphs: List,
                                      floor_names: List[str],
                                      floor_areas_list: List[Dict],
                                      start_floor_name: str,
                                      end_floor_name: str,
                                      pixel_to_meter_ratio: float = 0.1) -> Dict:
    """
    Çok katlı rotalar için alternatifler üretir
    Her kat segmenti için hem shortest hem least_turns hesaplar
    
    Returns:
        {
            'shortest': { segments, floors, ... },
            'least_turns': { segments, floors, ... }
        }
    """
    # Başlangıç noktası (en yakın portalı seçmek için)
    start_point = start_room.get('center') if start_room else None
    
    # Portal planını hesapla (her iki alternatif için aynı, başlangıca en yakın portal)
    floor_transitions = plan_floor_transitions(
        start_floor_name, end_floor_name, all_graphs, floor_names, start_point
    )
    
    if not floor_transitions:
        return {}
    
    alternatives = {}
    
    # Her alternatif için rotayı hesapla
    for route_type in ['shortest', 'least_turns']:
        cost_type = 'distance' if route_type == 'shortest' else 'turns'
        
        # Her kat segmenti için rota hesapla
        segments_by_floor = {}  # floor_name -> segment_data
        
        current_floor = start_floor_name
        current_graph = start_graph
        current_floor_areas = floor_areas_list[floor_names.index(start_floor_name)]
        
        # İlk segment: Başlangıç -> İlk Portal
        first_transition = floor_transitions[0]
        from_floor, to_floor, portal_data = first_transition
        
        first_portal_conn = portal_data['connection']
        start_conn_id = find_nearest_connection_to_room(start_room['center'], current_graph, room_id=start_room['id'])
        
        segment = create_route_segment(
            start_conn_id=start_conn_id,
            end_conn_id=first_portal_conn['id'],
            graph=current_graph,
            floor_areas=current_floor_areas,
            start_desc=f"{start_room['type']} - {start_room['id']}",
            end_desc=f"Portal",
            floor_name=current_floor,
            pixel_to_meter_ratio=pixel_to_meter_ratio,
            route_type=cost_type
        )
        
        if segment:
            segments_by_floor[current_floor] = segment
        
        # Ara katlar ve son kat
        for i, (from_floor, to_floor, portal_data) in enumerate(floor_transitions):
            to_floor_idx = floor_names.index(to_floor)
            to_graph = all_graphs[to_floor_idx]
            to_floor_areas = floor_areas_list[to_floor_idx]
            
            portal_info = portal_data['portal_info']
            matching_portal = find_matching_portal(
                portal_info,
                str(get_floor_number(from_floor)),
                to_graph.connections
            )
            
            if not matching_portal:
                continue
            
            # Son geçiş mi?
            if i == len(floor_transitions) - 1:
                # Hedef odaya git (önce odanın kendi kapısını ara)
                end_conn_id = find_nearest_connection_to_room(end_room['center'], to_graph, room_id=end_room['id'])
                
                segment = create_route_segment(
                    start_conn_id=matching_portal['id'],
                    end_conn_id=end_conn_id,
                    graph=to_graph,
                    floor_areas=to_floor_areas,
                    start_desc="Portal",
                    end_desc=f"{end_room['type']} - {end_room['id']}",
                    floor_name=to_floor,
                    pixel_to_meter_ratio=pixel_to_meter_ratio,
                    route_type=cost_type
                )
                
                if segment:
                    segments_by_floor[to_floor] = segment
            else:
                # Ara kat, sonraki portala git
                next_transition = floor_transitions[i + 1]
                next_portal_data = next_transition[2]
                next_portal_conn = next_portal_data['connection']
                
                segment = create_route_segment(
                    start_conn_id=matching_portal['id'],
                    end_conn_id=next_portal_conn['id'],
                    graph=to_graph,
                    floor_areas=to_floor_areas,
                    start_desc="Portal",
                    end_desc="Portal",
                    floor_name=to_floor,
                    pixel_to_meter_ratio=pixel_to_meter_ratio,
                    route_type=cost_type
                )
                
                if segment:
                    segments_by_floor[to_floor] = segment
        
        alternatives[route_type] = {
            'segments': segments_by_floor,
            'route_type': route_type,
            'name': 'En Kısa Mesafe' if route_type == 'shortest' else 'En Az Dönüş'
        }
    
    return alternatives



