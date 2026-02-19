"""
Route Sampler - Çeşitli senaryoları temsil eden rotaları seçer
İnsan tarafından oluşturulacak rota tarifleri için örnek veri seti hazırlar
"""
import json
import os
from typing import List, Dict, Tuple, Optional
import random
import pandas as pd


class RouteSampler:
    """
    Tüm rotalardan stratified sampling ile çeşitli örnekler seçer
    Farklı özelliklere sahip rotaları dengeli şekilde örnekler
    """
    
    def __init__(self, routes_data: Dict):
        """
        Args:
            routes_data: generate_all_routes_for_floor() çıktısı
        """
        self.routes_data = routes_data
        self.all_routes = list(routes_data['routes'].items())
        self.selected_routes = []
    
    def categorize_routes(self) -> Dict[str, List]:
        """
        Rotaları farklı kategorilere ayırır
        
        Kategoriler:
        - Mesafe: çok kısa, kısa, orta, uzun, çok uzun
        - Dönüş sayısı: az, orta, çok
        - Karmaşıklık: basit, orta, karmaşık
        - Oda tipleri: Shop->Food, Food->Other, vb.
        - Kat geçişleri: aynı kat, 1 geçiş, çok geçiş
        """
        categories = {
            # Mesafe kategorileri
            'very_short': [],      # < 10m
            'short': [],           # 10-30m
            'medium': [],          # 30-60m
            'long': [],            # 60-100m
            'very_long': [],       # > 100m
            
            # Dönüş sayısı kategorileri
            'few_turns': [],       # 0-2 dönüş
            'medium_turns': [],    # 3-5 dönüş
            'many_turns': [],      # 6+ dönüş
            
            # Karmaşıklık kategorileri (mesafe/adım oranı)
            'simple': [],          # Düz, az dönüşlü
            'moderate': [],        # Orta karmaşıklık
            'complex': [],         # Çok dönüşlü, karmaşık
            
            # Kat geçişi kategorileri
            'same_floor': [],      # Aynı kat
            'one_transition': [],  # 1 kat geçişi
            'two_transitions': [], # 2 kat geçişi
            'many_transitions': [],# 3+ kat geçişi
            
            # Oda tipi kombinasyonları
            'shop_to_shop': [],
            'shop_to_food': [],
            'shop_to_other': [],
            'shop_to_green': [],
            'food_to_food': [],
            'food_to_other': [],
            'food_to_green': [],
            'other_to_other': [],
            'other_to_green': [],
            'green_to_green': [],
        }
        
        for route_key, route_data in self.all_routes:
            distance = route_data['summary']['total_distance_meters']
            turns = route_data.get('turns_count', route_data['summary'].get('turns_count', 0))
            steps = route_data['summary']['total_steps']
            
            from_type = route_data['from']['type'].lower()
            to_type = route_data['to']['type'].lower()
            
            # Kat geçişi bilgisi (multi-floor desteği)
            is_multi_floor = route_data.get('is_multi_floor', False)
            floor_transitions = route_data.get('floor_transitions', 0)
            
            # Mesafe kategorileri
            if distance < 10:
                categories['very_short'].append((route_key, route_data))
            elif distance < 30:
                categories['short'].append((route_key, route_data))
            elif distance < 60:
                categories['medium'].append((route_key, route_data))
            elif distance < 100:
                categories['long'].append((route_key, route_data))
            else:
                categories['very_long'].append((route_key, route_data))
            
            # Dönüş sayısı kategorileri
            if turns <= 2:
                categories['few_turns'].append((route_key, route_data))
            elif turns <= 5:
                categories['medium_turns'].append((route_key, route_data))
            else:
                categories['many_turns'].append((route_key, route_data))
            
            # Karmaşıklık (dönüş/mesafe oranı)
            complexity = turns / max(distance, 1)  # Dönüş yoğunluğu
            if complexity < 0.05:
                categories['simple'].append((route_key, route_data))
            elif complexity < 0.15:
                categories['moderate'].append((route_key, route_data))
            else:
                categories['complex'].append((route_key, route_data))
            
            # Kat geçişi kategorileri
            if not is_multi_floor or floor_transitions == 0:
                categories['same_floor'].append((route_key, route_data))
            elif floor_transitions == 1:
                categories['one_transition'].append((route_key, route_data))
            elif floor_transitions == 2:
                categories['two_transitions'].append((route_key, route_data))
            else:
                categories['many_transitions'].append((route_key, route_data))
            
            # Oda tipi kombinasyonları
            combo_key = f"{from_type}_to_{to_type}"
            if combo_key in categories:
                categories[combo_key].append((route_key, route_data))
        
        return categories
    
    def calculate_route_diversity_score(self, route_data: Dict) -> float:
        """
        Rotanın çeşitlilik skorunu hesaplar (farklılık puanı)
        Yüksek skor = daha ilginç/farklı rota
        """
        distance = route_data['summary']['total_distance_meters']
        turns = route_data.get('turns_count', route_data['summary'].get('turns_count', 0))
        steps = route_data['summary']['total_steps']
        
        # Kat geçişi bilgisi
        is_multi_floor = route_data.get('is_multi_floor', False)
        floor_transitions = route_data.get('floor_transitions', 0)
        
        # Farklı faktörler
        score = 0.0
        
        # Kat geçişi çeşitliliği (ÖNEMLİ - daha yüksek ağırlık)
        if is_multi_floor:
            score += 3.0  # Katlar arası rotalar önemli
            if floor_transitions >= 2:
                score += 2.0  # Çok katlı geçişler daha ilginç
            if floor_transitions >= 3:
                score += 1.5  # Ekstrem durum
        
        # Mesafe çeşitliliği (orta mesafeler daha az ilginç)
        if distance < 15 or distance > 80:
            score += 2.0
        elif distance < 25 or distance > 60:
            score += 1.0
        
        # Dönüş çeşitliliği
        if turns == 0 or turns > 6:
            score += 2.0
        elif turns <= 2 or turns >= 5:
            score += 1.0
        
        # Adım sayısı
        if steps > 8:
            score += 1.0
        
        # Karmaşıklık oranı
        complexity = turns / max(distance, 1)
        if complexity > 0.15 or complexity < 0.03:
            score += 1.5
        
        return score
    
    def select_representative_sample(self, 
                                    sample_size: int = 50,
                                    ensure_diversity: bool = True,
                                    max_same_location_usage: int = 1) -> List[Tuple[str, Dict]]:
        """
        Stratified sampling ile temsili örneklem seçer
        
        Args:
            sample_size: Seçilecek rota sayısı
            ensure_diversity: Çeşitlilik garantisi (her kategoriden en az 1)
            max_same_location_usage: Her birim maksimum kaç kez kullanılabilir
        
        Returns:
            Seçilen rotalar listesi [(route_key, route_data), ...]
        """
        categories = self.categorize_routes()
        selected = []
        selected_keys = set()
        
        # Her birim ID'sinin kaç kez kullanıldığını takip et
        from_id_count = {}
        to_id_count = {}
        
        # Öncelik kategorileri - her birinden mutlaka al
        priority_categories = [
            'very_short', 'very_long',      # Ekstrem mesafeler
            'few_turns', 'many_turns',       # Ekstrem dönüş sayıları
            'simple', 'complex',              # Ekstrem karmaşıklıklar
            'one_transition', 'two_transitions', 'many_transitions',  # Katlar arası (ÖNEMLİ)
        ]
        
        if ensure_diversity:
            print("\nÇeşitlilik garantisi için öncelikli kategorilerden seçim yapılıyor...")
            
            # Her öncelikli kategoriden en az 1 tane al
            for cat in priority_categories:
                if categories[cat] and len(selected) < sample_size:
                    # En yüksek diversity score'a sahip olanı seç
                    sorted_routes = sorted(
                        categories[cat],
                        key=lambda x: self.calculate_route_diversity_score(x[1]),
                        reverse=True
                    )
                    
                    for route_key, route_data in sorted_routes:
                        if route_key not in selected_keys:
                            # Birim kullanım kontrolü
                            from_id = route_data['from']['id']
                            to_id = route_data['to']['id']
                            
                            from_count = from_id_count.get(from_id, 0)
                            to_count = to_id_count.get(to_id, 0)
                            
                            # Her iki birim de limit altındaysa seç
                            if from_count < max_same_location_usage and to_count < max_same_location_usage:
                                selected.append((route_key, route_data))
                                selected_keys.add(route_key)
                                from_id_count[from_id] = from_count + 1
                                to_id_count[to_id] = to_count + 1
                                print(f"  ✓ {cat}: {route_key} (skor: {self.calculate_route_diversity_score(route_data):.2f})")
                                break
        
        # Kalan kontenjanı tüm kategorilerden dengeli dağıt
        remaining = sample_size - len(selected)
        
        if remaining > 0:
            print(f"\nKalan {remaining} rota için dengeli dağılım yapılıyor...")
            
            # Oda tipi kombinasyonlarından seç
            room_type_categories = [k for k in categories.keys() if '_to_' in k]
            
            # Her oda tipi kombinasyonundan dengeli say
            routes_per_combo = max(1, remaining // len(room_type_categories))
            
            for combo_cat in room_type_categories:
                if len(selected) >= sample_size:
                    break
                
                if categories[combo_cat]:
                    # Diversity score'a göre sırala
                    sorted_routes = sorted(
                        categories[combo_cat],
                        key=lambda x: self.calculate_route_diversity_score(x[1]),
                        reverse=True
                    )
                    
                    added_count = 0
                    for route_key, route_data in sorted_routes:
                        if route_key not in selected_keys and added_count < routes_per_combo:
                            # Birim kullanım kontrolü
                            from_id = route_data['from']['id']
                            to_id = route_data['to']['id']
                            
                            from_count = from_id_count.get(from_id, 0)
                            to_count = to_id_count.get(to_id, 0)
                            
                            # Her iki birim de limit altındaysa seç
                            if from_count < max_same_location_usage and to_count < max_same_location_usage:
                                selected.append((route_key, route_data))
                                selected_keys.add(route_key)
                                from_id_count[from_id] = from_count + 1
                                to_id_count[to_id] = to_count + 1
                                added_count += 1
                                
                                if len(selected) >= sample_size:
                                    break
        
        # Hala eksikse, kalan en yüksek skorlu rotaları ekle
        if len(selected) < sample_size:
            print(f"\nEksik kontenjan için en yüksek skorlu rotalar ekleniyor...")
            remaining_routes = [
                (k, d) for k, d in self.all_routes 
                if k not in selected_keys
            ]
            
            sorted_remaining = sorted(
                remaining_routes,
                key=lambda x: self.calculate_route_diversity_score(x[1]),
                reverse=True
            )
            
            for route_key, route_data in sorted_remaining:
                if len(selected) >= sample_size:
                    break
                
                # Birim kullanım kontrolü
                from_id = route_data['from']['id']
                to_id = route_data['to']['id']
                
                from_count = from_id_count.get(from_id, 0)
                to_count = to_id_count.get(to_id, 0)
                
                # Her iki birim de limit altındaysa seç
                if from_count < max_same_location_usage and to_count < max_same_location_usage:
                    selected.append((route_key, route_data))
                    selected_keys.add(route_key)
                    from_id_count[from_id] = from_count + 1
                    to_id_count[to_id] = to_count + 1
        
        self.selected_routes = selected
        
        # İstatistik: En çok kullanılan birimler
        print(f"\nBirim kullanım istatistikleri:")
        print(f"  Toplam farklı başlangıç birimi: {len(from_id_count)}")
        print(f"  Toplam farklı bitiş birimi: {len(to_id_count)}")
        print(f"  En çok kullanılan başlangıç: {max(from_id_count.values()) if from_id_count else 0} kez")
        print(f"  En çok kullanılan bitiş: {max(to_id_count.values()) if to_id_count else 0} kez")
        
        return selected
    
    def generate_statistics(self) -> Dict:
        """Seçilen rotalar için istatistikler üretir"""
        if not self.selected_routes:
            return {}
        
        distances = [r[1]['summary']['total_distance_meters'] for r in self.selected_routes]
        turns = [r[1].get('turns_count', r[1]['summary'].get('turns_count', 0)) for r in self.selected_routes]
        steps = [r[1]['summary']['total_steps'] for r in self.selected_routes]
        floor_transitions = [r[1].get('floor_transitions', 0) for r in self.selected_routes]
        
        # Oda tipi kombinasyonları
        room_combos = {}
        for route_key, route_data in self.selected_routes:
            from_type = route_data['from']['type']
            to_type = route_data['to']['type']
            combo = f"{from_type}->{to_type}"
            room_combos[combo] = room_combos.get(combo, 0) + 1
        
        # Kat geçişi dağılımı
        floor_transition_dist = {
            'same_floor': sum(1 for ft in floor_transitions if ft == 0),
            'one_transition': sum(1 for ft in floor_transitions if ft == 1),
            'two_transitions': sum(1 for ft in floor_transitions if ft == 2),
            'many_transitions': sum(1 for ft in floor_transitions if ft >= 3)
        }
        
        return {
            'total_selected': len(self.selected_routes),
            'distance': {
                'min': min(distances),
                'max': max(distances),
                'avg': sum(distances) / len(distances),
                'median': sorted(distances)[len(distances) // 2]
            },
            'turns': {
                'min': min(turns),
                'max': max(turns),
                'avg': sum(turns) / len(turns),
                'median': sorted(turns)[len(turns) // 2]
            },
            'steps': {
                'min': min(steps),
                'max': max(steps),
                'avg': sum(steps) / len(steps),
                'median': sorted(steps)[len(steps) // 2]
            },
            'floor_transitions': {
                'min': min(floor_transitions),
                'max': max(floor_transitions),
                'avg': sum(floor_transitions) / len(floor_transitions),
                'distribution': floor_transition_dist
            },
            'room_type_combinations': room_combos
        }
    
    def export_sample(self, output_file: str = "route_sample.json"):
        """Seçilen rotaları JSON dosyasına kaydeder"""
        if not self.selected_routes:
            print("Henüz rota seçilmedi. Önce select_representative_sample() çalıştırın.")
            return
        
        # Seçilen rotaları organize et
        sample_data = {
            'metadata': {
                'floor': self.routes_data['floor'],
                'total_available_routes': len(self.all_routes),
                'sample_size': len(self.selected_routes),
                'sampling_method': 'stratified_diversity_based'
            },
            'statistics': self.generate_statistics(),
            'routes': {}
        }
        
        # Her rotayı ekle
        for i, (route_key, route_data) in enumerate(self.selected_routes, 1):
            sample_data['routes'][route_key] = {
                'sample_id': i,
                'diversity_score': self.calculate_route_diversity_score(route_data),
                **route_data
            }
        
        # JSON'a kaydet
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ Örneklem {output_file} dosyasına kaydedildi.")
        return output_file
    
    def print_sample_summary(self):
        """Seçilen rotaların özetini yazdırır"""
        if not self.selected_routes:
            print("Henüz rota seçilmedi.")
            return
        
        stats = self.generate_statistics()
        
        print("\n" + "="*70)
        print("ÖRNEKLEM ÖZETİ")
        print("="*70)
        print(f"Toplam Seçilen Rota: {stats['total_selected']}")
        print(f"Toplam Mevcut Rota: {len(self.all_routes)}")
        print(f"Örneklem Oranı: {(stats['total_selected'] / len(self.all_routes) * 100):.1f}%")
        
        print("\nMESAFE İSTATİSTİKLERİ:")
        print(f"  En Kısa: {stats['distance']['min']:.1f}m")
        print(f"  En Uzun: {stats['distance']['max']:.1f}m")
        print(f"  Ortalama: {stats['distance']['avg']:.1f}m")
        print(f"  Medyan: {stats['distance']['median']:.1f}m")
        
        print("\nDÖNÜŞ İSTATİSTİKLERİ:")
        print(f"  En Az: {stats['turns']['min']}")
        print(f"  En Çok: {stats['turns']['max']}")
        print(f"  Ortalama: {stats['turns']['avg']:.1f}")
        print(f"  Medyan: {stats['turns']['median']}")
        
        print("\nADIM İSTATİSTİKLERİ:")
        print(f"  En Az: {stats['steps']['min']}")
        print(f"  En Çok: {stats['steps']['max']}")
        print(f"  Ortalama: {stats['steps']['avg']:.1f}")
        print(f"  Medyan: {stats['steps']['median']}")
        
        print("\nKAT GEÇİŞİ İSTATİSTİKLERİ:")
        print(f"  En Az: {stats['floor_transitions']['min']}")
        print(f"  En Çok: {stats['floor_transitions']['max']}")
        print(f"  Ortalama: {stats['floor_transitions']['avg']:.1f}")
        print(f"  Dağılım:")
        for transition_type, count in stats['floor_transitions']['distribution'].items():
            print(f"    {transition_type}: {count} rota")
        
        print("\nODA TİPİ KOMBİNASYONLARI:")
        for combo, count in sorted(stats['room_type_combinations'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {combo}: {count} rota")
        
        print("="*70)
    
    def export_for_human_annotation(self, output_dir: str = "human_annotation_tasks",
                                    excel_path: str = "Zorlu Center List.xlsx"):
        """
        İnsan annotasyonu için görev dosyaları oluşturur
        Her rota için ayrı bir dosya veya grup halinde
        
        Args:
            output_dir: Çıktı dizini
            excel_path: ID-Title eşleştirme Excel dosyası
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Excel dosyasından ID-Title mapping'i yükle
        id_to_title = self._load_id_title_mapping(excel_path)
        
        tasks = []
        for i, (route_key, route_data) in enumerate(self.selected_routes, 1):
            is_multi_floor = route_data.get('is_multi_floor', False)
            floor_transitions = route_data.get('floor_transitions', 0)
            
            # Başlangıç ve bitiş için ID'leri al
            from_id = route_data['from']['id']
            to_id = route_data['to']['id']
            
            # Title'ları bul
            from_title = id_to_title.get(from_id, "")
            to_title = id_to_title.get(to_id, "")
            
            # Kat bilgilerini al (route_key'den parse et veya route_data'dan)
            from_floor = self._extract_floor_from_route_key(route_key, 'from')
            to_floor = self._extract_floor_from_route_key(route_key, 'to')
            
            # Format: "Kat X / Grup - ID (Title)"
            from_full = f"{from_floor} / {route_data['from']['type']} - {from_id}"
            if from_title:
                from_full += f" ({from_title})"
            
            to_full = f"{to_floor} / {route_data['to']['type']} - {to_id}"
            if to_title:
                to_full += f" ({to_title})"
            
            task = {
                'task_id': i,
                'route_key': route_key,
                'from': from_full,
                'to': to_full,
                'is_multi_floor': is_multi_floor,
                'floor_transitions': floor_transitions,
                'turns_count': route_data.get('turns_count', route_data['summary'].get('turns_count', 0))
            }
            tasks.append(task)
        
        # Tek dosyaya kaydet
        output_file = os.path.join(output_dir, "annotation_tasks.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': {
                    'total_tasks': len(tasks),
                    'floor': self.routes_data['floor'],
                    'instructions': 'Her rota için "human_generated_directions" alanına doğal dille rota tarifi yazın.'
                },
                'tasks': tasks
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ İnsan annotasyon görevleri {output_file} dosyasına kaydedildi.")
        
        # Ayrıca basit metin formatında da kaydet (kolay okunabilir)
        text_file = os.path.join(output_dir, "annotation_tasks.txt")
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("İNSAN ROTA TARİFİ OLUŞTURMA GÖREVLERİ\n")
            f.write("="*70 + "\n\n")
            
            for task in tasks:
                f.write(f"\nGÖREV #{task['task_id']}\n")
                f.write("-"*70 + "\n")
                f.write(f"Başlangıç: {task['from']}\n")
                f.write(f"Hedef: {task['to']}\n")
                if task['is_multi_floor']:
                    f.write(f"Kat Geçişi: {task['floor_transitions']} kez\n")
                f.write(f"Dönüş Sayısı: {task['turns_count']}\n\n")
                
                f.write("Rota Tarifi:\n")
                f.write("  [Buraya doğal dille rota tarifi yazın]\n\n")
                f.write("="*70 + "\n")
        
        print(f"✓ Metin formatı {text_file} dosyasına kaydedildi.")
        
        return output_file, text_file
    
    def _load_id_title_mapping(self, excel_path: str) -> Dict[str, str]:
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
    
    def _extract_floor_from_route_key(self, route_key: str, position: str) -> str:
        """
        Route key'den kat bilgisini çıkarır
        
        Args:
            route_key: Örn: "Kat 0_Shop_ID043_to_Kat -1_Shop_ID-158"
            position: 'from' veya 'to'
        
        Returns:
            Kat ismi (örn: "Kat 0", "Kat -1")
        """
        try:
            if '_to_' in route_key:
                parts = route_key.split('_to_')
                
                if position == 'from':
                    # "Kat 0_Shop_ID043" -> "Kat 0"
                    from_part = parts[0]
                    # İlk underscore'a kadar olan kısım kat ismi
                    if from_part.startswith('Kat'):
                        # "Kat 0" veya "Kat -1" gibi
                        floor_parts = from_part.split('_')
                        if len(floor_parts) >= 2:
                            return f"{floor_parts[0]} {floor_parts[1]}"
                        return floor_parts[0]
                else:  # to
                    # "Kat -1_Shop_ID-158" -> "Kat -1"
                    to_part = parts[1]
                    if to_part.startswith('Kat'):
                        floor_parts = to_part.split('_')
                        if len(floor_parts) >= 2:
                            return f"{floor_parts[0]} {floor_parts[1]}"
                        return floor_parts[0]
        except Exception as e:
            print(f"Uyarı: Route key parse hatası: {str(e)}")
        
        return "Unknown"


def select_sample_routes_from_file(routes_json_path: str, 
                                   sample_size: int = 50,
                                   output_prefix: str = None) -> str:
    """
    JSON dosyasından rota örneklerini seçer
    
    Args:
        routes_json_path: Rota JSON dosyası yolu
        sample_size: Seçilecek rota sayısı
        output_prefix: Çıktı dosyası öneki
    
    Returns:
        Çıktı dosyası yolu
    """
    # JSON dosyasını yükle
    with open(routes_json_path, 'r', encoding='utf-8') as f:
        routes_data = json.load(f)
    
    # Dosya formatını kontrol et (tek kat vs çok kat)
    if 'floor' in routes_data:
        # Tek kat formatı
        floor_name = routes_data['floor']
        print(f"\n{floor_name} için {sample_size} rota seçiliyor...")
    elif 'metadata' in routes_data and 'floors' in routes_data['metadata']:
        # Çok katlı format
        floor_name = "Tüm Katlar"
        floors_list = routes_data['metadata']['floors']
        print(f"\n{floor_name} ({', '.join(floors_list)}) için {sample_size} rota seçiliyor...")
        
        # metadata'dan floor bilgisini kaldır, yoksa RouteSampler hata verir
        # RouteSampler'ın beklediği formata dönüştür
        if 'floor' not in routes_data:
            routes_data['floor'] = floor_name
    else:
        print("Uyarı: Dosya formatı tanınamadı, varsayılan isim kullanılıyor")
        floor_name = "Unknown"
        routes_data['floor'] = floor_name
    
    # RouteSampler oluştur
    sampler = RouteSampler(routes_data)
    
    # Temsili örneklem seç
    sampler.select_representative_sample(sample_size=sample_size, ensure_diversity=True)
    
    # Özet yazdır
    sampler.print_sample_summary()
    
    # Dosyaya kaydet
    if output_prefix is None:
        output_prefix = floor_name.replace(' ', '_')
    
    sample_file = sampler.export_sample(f"route_sample_{output_prefix}.json")
    annotation_files = sampler.export_for_human_annotation(f"human_annotation_{output_prefix}")
    
    return sample_file

