"""
Batch Route Generator Multi-Floor - Tüm birimler arası rotalar (tek kat + çok kat)
"""
import json
import os
from typing import List, Dict, Tuple, Optional
import pandas as pd
from helpers.batch_route_generator import (
    generate_route_for_room_pair, find_nearest_connection_to_room
)
from helpers.multi_floor_route_generator import generate_multi_floor_route
from helpers.portal_matcher import get_floor_number


def generate_all_routes_multi_floor(all_graphs: List,
                                    floor_names: List[str],
                                    floor_areas_list: List[Dict],
                                    output_dir: str = "routes_multi",
                                    pixel_to_meter_ratio: float = 0.1,
                                    max_routes_per_floor_pair: Optional[int] = None,
                                    include_same_floor: bool = True,
                                    excel_path: str = "Zorlu Center List.xlsx") -> Dict:
    """
    Tüm katlar arasındaki tüm birimler için rotalar oluşturur
    
    Args:
        all_graphs: Tüm katların graph'ları
        floor_names: Kat isimleri
        floor_areas_list: Her katın oda bilgileri
        output_dir: Çıktı dizini
        pixel_to_meter_ratio: Piksel-metre oranı
        max_routes_per_floor_pair: Her kat çifti için maksimum rota sayısı (test için)
        include_same_floor: Aynı kat içi rotaları da dahil et
    
    Returns:
        İstatistik bilgileri
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Excel dosyasından ID-Title mapping'i yükle
    id_to_title = load_id_title_mapping(excel_path)
    
    # Tüm odaları katalogla
    all_rooms_by_floor = {}
    
    for floor_name, floor_areas in zip(floor_names, floor_areas_list):
        rooms = []
        for room_type, room_list in floor_areas.items():
            for room in room_list:
                rooms.append({
                    'id': room['id'],
                    'type': room_type,
                    'center': room['center'],
                    'area': room['area'],
                    'floor': floor_name
                })
        all_rooms_by_floor[floor_name] = rooms
    
    print(f"\n{'='*70}")
    print("ÇOK KATLI ROTA OLUŞTURMA")
    print(f"{'='*70}")
    print(f"Toplam Kat Sayısı: {len(floor_names)}")
    for floor_name, rooms in all_rooms_by_floor.items():
        print(f"  {floor_name}: {len(rooms)} birim")
    print(f"{'='*70}\n")
    
    # İstatistikler
    stats = {
        'same_floor': {'successful': 0, 'failed': 0},
        'multi_floor': {'successful': 0, 'failed': 0, 'by_transition_count': {}},
        'total': {'successful': 0, 'failed': 0}
    }
    
    all_routes = {}
    route_count = 0
    
    # Her kat çifti için rotaları oluştur
    for start_floor_name in floor_names:
        start_floor_idx = floor_names.index(start_floor_name)
        start_graph = all_graphs[start_floor_idx]
        start_floor_areas = floor_areas_list[start_floor_idx]
        start_rooms = all_rooms_by_floor[start_floor_name]
        
        for end_floor_name in floor_names:
            end_floor_idx = floor_names.index(end_floor_name)
            end_graph = all_graphs[end_floor_idx]
            end_floor_areas = floor_areas_list[end_floor_idx]
            end_rooms = all_rooms_by_floor[end_floor_name]
            
            is_same_floor = (start_floor_name == end_floor_name)
            
            # Aynı kat ise ve dahil edilmeyecekse atla
            if is_same_floor and not include_same_floor:
                continue
            
            print(f"\n{'='*70}")
            print(f"ROTA ÇİFTİ: {start_floor_name} -> {end_floor_name}")
            print(f"{'='*70}")
            
            floor_pair_count = 0
            
            for start_room in start_rooms:
                for end_room in end_rooms:
                    # Aynı oda ise atla
                    if is_same_floor and start_room['id'] == end_room['id']:
                        continue
                    
                    # Maksimum limit kontrolü
                    if max_routes_per_floor_pair and floor_pair_count >= max_routes_per_floor_pair:
                        break
                    
                    route_key = f"{start_floor_name}_{start_room['type']}_{start_room['id']}_to_{end_floor_name}_{end_room['type']}_{end_room['id']}"
                    
                    print(f"Rota ({route_count + 1}): {start_floor_name}/{start_room['type']}-{start_room['id']} -> {end_floor_name}/{end_room['type']}-{end_room['id']}")
                    
                    if is_same_floor:
                        # Aynı kat içi rota
                        route_data = generate_route_for_room_pair(
                            start_room=start_room,
                            end_room=end_room,
                            graph=start_graph,
                            floor_areas=start_floor_areas,
                            pixel_to_meter_ratio=pixel_to_meter_ratio
                        )
                        
                        if route_data:
                            route_data['is_multi_floor'] = False
                            route_data['floor_transitions'] = 0
                            # Temizle ve title ekle
                            cleaned_route = clean_and_add_titles(route_data, id_to_title)
                            all_routes[route_key] = cleaned_route
                            stats['same_floor']['successful'] += 1
                            stats['total']['successful'] += 1
                            print(f"  ✓ Aynı kat - Mesafe: {route_data['summary']['total_distance_meters']:.1f}m")
                        else:
                            stats['same_floor']['failed'] += 1
                            stats['total']['failed'] += 1
                            print(f"  ✗ Rota bulunamadı")
                    else:
                        # Katlar arası rota
                        route_data = generate_multi_floor_route(
                            start_room=start_room,
                            end_room=end_room,
                            start_graph=start_graph,
                            end_graph=end_graph,
                            all_graphs=all_graphs,
                            floor_names=floor_names,
                            floor_areas_list=floor_areas_list,
                            start_floor_name=start_floor_name,
                            end_floor_name=end_floor_name,
                            pixel_to_meter_ratio=pixel_to_meter_ratio
                        )
                        
                        if route_data:
                            # Temizle ve title ekle
                            cleaned_route = clean_and_add_titles(route_data, id_to_title)
                            all_routes[route_key] = cleaned_route
                            transitions = route_data['floor_transitions']
                            
                            stats['multi_floor']['successful'] += 1
                            stats['total']['successful'] += 1
                            
                            if transitions not in stats['multi_floor']['by_transition_count']:
                                stats['multi_floor']['by_transition_count'][transitions] = 0
                            stats['multi_floor']['by_transition_count'][transitions] += 1
                            
                            print(f"  ✓ Çok katlı ({transitions} geçiş) - Mesafe: {route_data['summary']['total_distance_meters']:.1f}m")
                        else:
                            stats['multi_floor']['failed'] += 1
                            stats['total']['failed'] += 1
                            print(f"  ✗ Rota bulunamadı")
                    
                    route_count += 1
                    floor_pair_count += 1
                
                if max_routes_per_floor_pair and floor_pair_count >= max_routes_per_floor_pair:
                    break
    
    # Sonuçları kaydet
    output_file = os.path.join(output_dir, "routes_all_floors.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'floors': floor_names,
                'total_rooms': sum(len(rooms) for rooms in all_rooms_by_floor.values())
            },
            'statistics': stats,
            'routes': all_routes
        }, f, ensure_ascii=False, indent=2)
    
    # Özet yazdır
    print(f"\n{'='*70}")
    print("GENEL ÖZET - TÜM KATLAR")
    print(f"{'='*70}")
    print(f"Aynı Kat İçi:")
    print(f"  Başarılı: {stats['same_floor']['successful']}")
    print(f"  Başarısız: {stats['same_floor']['failed']}")
    
    print(f"\nKatlar Arası:")
    print(f"  Başarılı: {stats['multi_floor']['successful']}")
    print(f"  Başarısız: {stats['multi_floor']['failed']}")
    
    if stats['multi_floor']['by_transition_count']:
        print(f"\n  Geçiş Sayısına Göre:")
        for transitions, count in sorted(stats['multi_floor']['by_transition_count'].items()):
            print(f"    {transitions} geçiş: {count} rota")
    
    print(f"\nTOPLAM:")
    print(f"  Başarılı: {stats['total']['successful']}")
    print(f"  Başarısız: {stats['total']['failed']}")
    if stats['total']['successful'] + stats['total']['failed'] > 0:
        success_rate = stats['total']['successful'] / (stats['total']['successful'] + stats['total']['failed']) * 100
        print(f"  Başarı Oranı: {success_rate:.1f}%")
    
    print(f"\nDosya: {output_file}")
    print(f"{'='*70}")
    
    return stats


def load_id_title_mapping(excel_path: str) -> Dict[str, str]:
    """
    Excel dosyasından ID-Title mapping'i yükler
    
    Args:
        excel_path: Excel dosyası yolu
    
    Returns:
        {ID: Title} dictionary
    """
    id_to_title = {}
    
    try:
        # Excel dosyasını oku
        df = pd.read_excel(excel_path)
        
        # Sütun isimlerini kontrol et
        if 'ID' in df.columns and 'Title' in df.columns:
            for _, row in df.iterrows():
                id_val = str(row['ID']).strip()
                title_val = str(row['Title']).strip() if pd.notna(row['Title']) else ""
                
                if id_val and id_val != 'nan':
                    id_to_title[id_val] = title_val
            
            print(f"✓ Excel'den {len(id_to_title)} adet ID-Title eşleştirmesi yüklendi.")
        else:
            print(f"Uyarı: Excel dosyasında 'ID' veya 'Title' sütunu bulunamadı.")
            print(f"Mevcut sütunlar: {list(df.columns)}")
    
    except FileNotFoundError:
        print(f"Uyarı: '{excel_path}' dosyası bulunamadı. Title'lar olmadan devam ediliyor.")
    except Exception as e:
        print(f"Uyarı: Excel dosyası okunurken hata: {str(e)}")
    
    return id_to_title


def clean_and_add_titles(route_data: Dict, id_to_title: Dict[str, str]) -> Dict:
    """
    Rota verisinden gereksiz alanları kaldırır ve title'ları ekler
    
    Args:
        route_data: Rota verisi
        id_to_title: ID-Title mapping
    
    Returns:
        Temizlenmiş ve title eklenmiş rota verisi
    """
    cleaned = {}
    
    # From bilgisi
    from_id = route_data['from']['id']
    from_title = id_to_title.get(from_id, "")
    cleaned['from'] = {
        'type': route_data['from']['type'],
        'id': from_id,
        'title': from_title,
        'center': route_data['from']['center']
    }
    
    # To bilgisi
    to_id = route_data['to']['id']
    to_title = id_to_title.get(to_id, "")
    cleaned['to'] = {
        'type': route_data['to']['type'],
        'id': to_id,
        'title': to_title,
        'center': route_data['to']['center']
    }
    
    # Multi-floor bilgisi
    cleaned['is_multi_floor'] = route_data.get('is_multi_floor', False)
    cleaned['floor_transitions'] = route_data.get('floor_transitions', 0)
    
    # Summary - sadece bazı alanlar
    cleaned['summary'] = {
        'total_steps': route_data['summary']['total_steps'],
        'turns_count': route_data['summary'].get('turns_count', 0)
    }
    
    if cleaned['is_multi_floor']:
        cleaned['summary']['floor_changes'] = route_data['summary'].get('floor_changes', 0)
    
    # Steps - sadece gerekli alanlar
    cleaned['steps'] = []
    for step in route_data['steps']:
        cleaned_step = {
            'step_number': step['step_number'],
            'action': step['action']
        }
        
        # Floor change için ek bilgiler
        if step.get('action') == 'FLOOR_CHANGE':
            cleaned_step['from_floor'] = step.get('from_floor')
            cleaned_step['to_floor'] = step.get('to_floor')
            cleaned_step['portal_type'] = step.get('portal_type')
        
        # Landmark varsa ekle
        if step.get('landmark'):
            cleaned_step['landmark'] = step['landmark']
        
        # Direction varsa ekle
        if step.get('direction'):
            cleaned_step['direction'] = step['direction']
        
        cleaned['steps'].append(cleaned_step)
    
    # Path connections
    if 'path_connections' in route_data:
        cleaned['path_connections'] = route_data['path_connections']
    
    # Turns count
    if 'turns_count' in route_data:
        cleaned['turns_count'] = route_data['turns_count']
    
    return cleaned


