"""
Route Visualizer - SVG üzerinde hesaplanan rotaları görselleştirir
Rota connection'larının rengini değiştirerek highlight eder
"""
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional
import os
import webbrowser


class RouteVisualizer:
    """SVG haritaları üzerinde rota görselleştirme - mevcut line'ları renklendirir"""
    
    def __init__(self, svg_file_path: str):
        """
        Args:
            svg_file_path: SVG dosyasının yolu
        """
        self.svg_file_path = svg_file_path
        self.tree = ET.parse(svg_file_path)
        self.root = self.tree.getroot()
        
        # Namespace'i belirle
        if '}' in self.root.tag:
            self.namespace = {'svg': self.root.tag.split('}')[0].strip('{')}
        else:
            self.namespace = {'svg': 'http://www.w3.org/2000/svg'}
        
        ET.register_namespace('', self.namespace['svg'])
    
    def highlight_route_connections(self, 
                                    connection_ids: List[str],
                                    route_color: str = "#FF0000",
                                    route_width: float = 3.0) -> int:
        """
        Rotayı oluşturan connection'ların rengini değiştirir
        
        Args:
            connection_ids: Rota connection ID'leri listesi
            route_color: Rota çizgisi rengi (hex)
            route_width: Rota çizgisi kalınlığı
        
        Returns:
            Renklendirilmiş connection sayısı
        """
        if not connection_ids:
            print("Uyarı: Connection ID listesi boş")
            return 0
        
        highlighted_count = 0
        
        # Paths grubunu bul
        paths_group = self.root.find(".//svg:g[@id='Paths']", self.namespace)
        if paths_group is None:
            print("Uyarı: 'Paths' grubu bulunamadı")
            return 0
        
        # Paths grubunu görünür hale getir
        self._make_paths_visible(paths_group)
        
        # Her connection ID için ilgili line/path'i bul ve rengini değiştir
        for conn_id in connection_ids:
            # Line elemanlarında ara
            line = paths_group.find(f".//svg:line[@id='{conn_id}']", self.namespace)
            if line is not None:
                line.set('stroke', route_color)
                line.set('stroke-width', str(route_width))
                line.set('stroke-linecap', 'round')
                highlighted_count += 1
            else:
                # Path elemanlarında ara
                path = paths_group.find(f".//svg:path[@id='{conn_id}']", self.namespace)
                if path is not None:
                    path.set('stroke', route_color)
                    path.set('stroke-width', str(route_width))
                    path.set('stroke-linecap', 'round')
                    highlighted_count += 1
        
        return highlighted_count
    
    def highlight_anchor_rooms(self, 
                               anchor_ids: List[str],
                               anchor_color: str = "#FFD700",
                               stroke_width: float = 3.0,
                               fill_opacity: float = 0.3) -> int:
        """
        Dönüş noktalarındaki anchor odaları highlight eder
        
        Args:
            anchor_ids: Anchor oda ID'leri listesi
            anchor_color: Anchor rengi (varsayılan: altın sarısı)
            stroke_width: Kenar çizgi kalınlığı
            fill_opacity: Dolgu şeffaflığı (0-1)
        
        Returns:
            Highlight edilen anchor sayısı
        """
        if not anchor_ids:
            return 0
        
        highlighted_count = 0
        
        # Rooms grubunu bul
        rooms_group = self.root.find(".//svg:g[@id='Rooms']", self.namespace)
        if rooms_group is None:
            # inkscape:label ile dene
            for g in self.root.findall(".//svg:g", self.namespace):
                label = g.get('{http://www.inkscape.org/namespaces/inkscape}label')
                if label == 'Rooms':
                    rooms_group = g
                    break
        
        # Portals grubunu da bul (yapısal elementler için)
        portals_group = self.root.find(".//svg:g[@id='Portals']", self.namespace)
        
        # Doors grubunu da bul (carpark door'ları için)
        doors_group = self.root.find(".//svg:g[@id='Doors']", self.namespace)
        
        # Doors grubunu görünür yap (varsayılan olarak invisible olabilir)
        if doors_group is not None:
            self._make_group_visible(doors_group)
        
        # Portals grubunu görünür yap (varsayılan olarak invisible olabilir)
        if portals_group is not None:
            self._make_group_visible(portals_group)
        
        # Icons grubunu gizle (portal'ların görünmesini engelliyor)
        icons_group = self.root.find(".//svg:g[@id='Icons']", self.namespace)
        if icons_group is not None:
            self._hide_group(icons_group)
        
        if rooms_group is None and portals_group is None and doors_group is None:
            print("Uyarı: Ne 'Rooms' ne 'Portals' ne de 'Doors' grubu bulunamadı")
            return 0
        
        print(f"  [Anchor Highlight] {len(anchor_ids)} anchor highlight edilecek: {list(set(anchor_ids))[:5]}...")
        
        # Her anchor ID için ilgili elementi bul ve highlight et
        for anchor_id in anchor_ids:
            element = None
            is_door = False
            
            # Önce Rooms grubunda ara
            if rooms_group is not None:
                element = rooms_group.find(f".//svg:path[@id='{anchor_id}']", self.namespace)
                
                if element is None:
                    element = rooms_group.find(f".//svg:rect[@id='{anchor_id}']", self.namespace)
                
                if element is None:
                    element = rooms_group.find(f".//*[@id='{anchor_id}']", self.namespace)
            
            # Rooms'da bulunamazsa Portals grubunda ara (Elev, Stairs gibi)
            if element is None and portals_group is not None:
                element = portals_group.find(f".//*[@id='{anchor_id}']", self.namespace)
            
            # Portals'da bulunamazsa Doors grubunda ara (carpark door'ları için)
            if element is None and doors_group is not None:
                element = doors_group.find(f".//svg:path[@id='{anchor_id}']", self.namespace)
                if element is None:
                    element = doors_group.find(f".//svg:line[@id='{anchor_id}']", self.namespace)
                if element is None:
                    element = doors_group.find(f".//*[@id='{anchor_id}']", self.namespace)
                if element is not None:
                    is_door = True
            
            # Hala bulunamazsa tüm SVG'de ara
            if element is None:
                element = self.root.find(f".//*[@id='{anchor_id}']", self.namespace)
            
            if element is not None:
                # Mevcut style'ı al
                current_style = element.get('style', '')
                
                # Door ise önce görünür yap (display ve visibility)
                if is_door or anchor_id.startswith('carpark-'):
                    # display:none ve visibility:hidden'ı kaldır
                    current_style = current_style.replace('display:none', 'display:inline')
                    current_style = current_style.replace('display: none', 'display:inline')
                    current_style = current_style.replace('visibility:hidden', 'visibility:visible')
                    current_style = current_style.replace('visibility: hidden', 'visibility:visible')
                    
                    # Yeni style oluştur (door'lar için farklı renk - turuncu)
                    new_style = f"display:inline;visibility:visible;stroke:{anchor_color};stroke-width:{stroke_width + 2};stroke-opacity:1;"
                else:
                    # Yeni style oluştur (normal odalar için)
                    new_style = f"fill:{anchor_color};fill-opacity:{fill_opacity};stroke:{anchor_color};stroke-width:{stroke_width};"
                
                # Eski fill ve stroke değerlerini kaldır, yenileri ekle
                style_parts = [s for s in current_style.split(';') if s.strip() and 
                              not s.strip().startswith('fill') and 
                              not s.strip().startswith('stroke') and
                              not s.strip().startswith('display') and
                              not s.strip().startswith('visibility')]
                style_parts.append(new_style)
                
                element.set('style', ';'.join(style_parts))
                highlighted_count += 1
            else:
                print(f"    [Anchor Highlight] '{anchor_id}' bulunamadı!")
        
        if highlighted_count > 0:
            print(f"  [Anchor Highlight] {highlighted_count} anchor highlight edildi")
        else:
            print(f"  [Anchor Highlight] Hiçbir anchor bulunamadı!")
        
        return highlighted_count
    
    def _make_paths_visible(self, paths_group):
        """Paths grubunu görünür hale getirir"""
        self._make_group_visible(paths_group)
    
    def _make_group_visible(self, group):
        """Herhangi bir grubu görünür hale getirir"""
        # Style attribute'ünü kontrol et ve görünür yap
        style = group.get('style', '')
        
        # display:none veya visibility:hidden gibi gizleme ifadelerini kaldır/değiştir
        if 'display' in style:
            # display:none'u display:inline ile değiştir
            style = style.replace('display:none', 'display:inline')
            style = style.replace('display: none', 'display:inline')
        
        if 'visibility' in style:
            # visibility:hidden'ı visibility:visible ile değiştir
            style = style.replace('visibility:hidden', 'visibility:visible')
            style = style.replace('visibility: hidden', 'visibility:visible')
        
        # opacity:0 varsa düzelt
        if 'opacity' in style:
            style = style.replace('opacity:0', 'opacity:1')
            style = style.replace('opacity: 0', 'opacity:1')
        
        # Eğer hala display yok ise, açıkça ekle
        if 'display' not in style:
            if style and not style.endswith(';'):
                style += ';'
            style += 'display:inline;'
        
        # Güncellenen style'ı uygula
        group.set('style', style)
        
        # display attribute'ü varsa inline yap
        if 'display' in group.attrib:
            group.set('display', 'inline')
        
        # visibility attribute'ü varsa düzelt
        if 'visibility' in group.attrib:
            group.set('visibility', 'visible')
        
        # opacity attribute'ü varsa düzelt
        if 'opacity' in group.attrib:
            group.set('opacity', '1')
    
    def _hide_group(self, group):
        """Bir grubu gizler"""
        # Style attribute'ünü al veya oluştur
        style = group.get('style', '')
        
        # display:none ekle
        if 'display' in style:
            # Mevcut display değerini none ile değiştir
            import re
            style = re.sub(r'display\s*:\s*\w+', 'display:none', style)
        else:
            if style and not style.endswith(';'):
                style += ';'
            style += 'display:none;'
        
        # Güncellenen style'ı uygula
        group.set('style', style)
        
        # display attribute'ü varsa none yap
        if 'display' in group.attrib:
            group.set('display', 'none')
    
    def draw_direction_arrow(self, 
                             start_point: Tuple[float, float],
                             direction_vector: Tuple[float, float],
                             arrow_length: float = 120.0,
                             arrow_color: str = "#00FF00",
                             arrow_width: float = 6.0,
                             label: str = None,
                             show_confidence: bool = True,
                             confidence: float = 1.0,
                             compass: str = None):
        """
        Başlangıç noktasından yön okunu çizer.
        
        Args:
            start_point: Ok başlangıç noktası (x, y)
            direction_vector: Normalize edilmiş yön vektörü (dx, dy)
            arrow_length: Ok uzunluğu
            arrow_color: Ok rengi
            arrow_width: Ok kalınlığı
            label: Gösterilecek etiket (None ise compass yönü gösterilir)
            show_confidence: Güven değerini gösterip göstermeme
            confidence: Güven değeri (0-1)
            compass: Pusula yönü metni
        """
        import math
        
        dx, dy = direction_vector
        end_x = start_point[0] + dx * arrow_length
        end_y = start_point[1] + dy * arrow_length
        
        # Ana SVG namespace'i
        svg_ns = self.namespace['svg']
        
        # Arrow grubu oluştur
        arrow_group = ET.SubElement(self.root, f'{{{svg_ns}}}g')
        arrow_group.set('id', 'DirectionArrow')
        arrow_group.set('style', 'display:inline')
        
        # Defs elementi - arrowhead marker için
        defs = self.root.find(f'.//{{{svg_ns}}}defs')
        if defs is None:
            defs = ET.SubElement(self.root, f'{{{svg_ns}}}defs')
        
        # Arrowhead marker
        marker_id = 'direction-arrowhead'
        marker = ET.SubElement(defs, f'{{{svg_ns}}}marker')
        marker.set('id', marker_id)
        marker.set('markerWidth', '10')
        marker.set('markerHeight', '7')
        marker.set('refX', '9')
        marker.set('refY', '3.5')
        marker.set('orient', 'auto')
        marker.set('markerUnits', 'strokeWidth')
        
        # Arrowhead path (üçgen)
        arrow_path = ET.SubElement(marker, f'{{{svg_ns}}}polygon')
        arrow_path.set('points', '0 0, 10 3.5, 0 7')
        arrow_path.set('fill', arrow_color)
        
        # Glow effect için filter
        filter_elem = ET.SubElement(defs, f'{{{svg_ns}}}filter')
        filter_elem.set('id', 'arrow-glow')
        filter_elem.set('x', '-50%')
        filter_elem.set('y', '-50%')
        filter_elem.set('width', '200%')
        filter_elem.set('height', '200%')
        
        fe_gaussian = ET.SubElement(filter_elem, f'{{{svg_ns}}}feGaussianBlur')
        fe_gaussian.set('stdDeviation', '3')
        fe_gaussian.set('result', 'coloredBlur')
        
        fe_merge = ET.SubElement(filter_elem, f'{{{svg_ns}}}feMerge')
        fe_merge_node1 = ET.SubElement(fe_merge, f'{{{svg_ns}}}feMergeNode')
        fe_merge_node1.set('in', 'coloredBlur')
        fe_merge_node2 = ET.SubElement(fe_merge, f'{{{svg_ns}}}feMergeNode')
        fe_merge_node2.set('in', 'SourceGraphic')
        
        # Glow çizgisi (arka plan)
        glow_line = ET.SubElement(arrow_group, f'{{{svg_ns}}}line')
        glow_line.set('x1', str(start_point[0]))
        glow_line.set('y1', str(start_point[1]))
        glow_line.set('x2', str(end_x))
        glow_line.set('y2', str(end_y))
        glow_line.set('stroke', arrow_color)
        glow_line.set('stroke-width', str(arrow_width + 4))
        glow_line.set('stroke-opacity', '0.4')
        glow_line.set('filter', 'url(#arrow-glow)')
        
        # Ana ok çizgisi
        main_line = ET.SubElement(arrow_group, f'{{{svg_ns}}}line')
        main_line.set('x1', str(start_point[0]))
        main_line.set('y1', str(start_point[1]))
        main_line.set('x2', str(end_x))
        main_line.set('y2', str(end_y))
        main_line.set('stroke', arrow_color)
        main_line.set('stroke-width', str(arrow_width))
        main_line.set('marker-end', f'url(#{marker_id})')
        main_line.set('stroke-linecap', 'round')
        
        # Başlangıç noktası dairesi
        start_circle = ET.SubElement(arrow_group, f'{{{svg_ns}}}circle')
        start_circle.set('cx', str(start_point[0]))
        start_circle.set('cy', str(start_point[1]))
        start_circle.set('r', str(arrow_width + 4))
        start_circle.set('fill', arrow_color)
        start_circle.set('stroke', '#FFFFFF')
        start_circle.set('stroke-width', '2')
        
        # İç daire (pulse efekti için)
        inner_circle = ET.SubElement(arrow_group, f'{{{svg_ns}}}circle')
        inner_circle.set('cx', str(start_point[0]))
        inner_circle.set('cy', str(start_point[1]))
        inner_circle.set('r', str(arrow_width))
        inner_circle.set('fill', '#FFFFFF')
        
        # Etiket metni
        label_text = label if label else (compass if compass else "İleri")
        
        # Metin arka planı (okunabilirlik için)
        text_x = end_x + dx * 20
        text_y = end_y + dy * 20
        
        # Metin arka plan kutusu
        text_bg = ET.SubElement(arrow_group, f'{{{svg_ns}}}rect')
        text_bg.set('x', str(text_x - 5))
        text_bg.set('y', str(text_y - 20))
        text_bg.set('width', '120')
        text_bg.set('height', '50' if show_confidence else '30')
        text_bg.set('rx', '5')
        text_bg.set('fill', 'rgba(0,0,0,0.7)')
        text_bg.set('stroke', arrow_color)
        text_bg.set('stroke-width', '2')
        
        # Yön metni
        text_elem = ET.SubElement(arrow_group, f'{{{svg_ns}}}text')
        text_elem.set('x', str(text_x + 55))
        text_elem.set('y', str(text_y))
        text_elem.set('fill', '#FFFFFF')
        text_elem.set('font-family', 'Arial, sans-serif')
        text_elem.set('font-size', '16')
        text_elem.set('font-weight', 'bold')
        text_elem.set('text-anchor', 'middle')
        text_elem.text = f"→ {label_text}"
        
        # Güven değeri
        if show_confidence:
            conf_text = ET.SubElement(arrow_group, f'{{{svg_ns}}}text')
            conf_text.set('x', str(text_x + 55))
            conf_text.set('y', str(text_y + 20))
            conf_text.set('fill', arrow_color)
            conf_text.set('font-family', 'Arial, sans-serif')
            conf_text.set('font-size', '12')
            conf_text.set('text-anchor', 'middle')
            conf_text.text = f"Güven: {confidence:.0%}"
        
        print(f"  [Direction Arrow] Ok çizildi: {label_text} ({confidence:.0%} güven)")
        return True
    
    def save_and_open(self, output_path: str, auto_open: bool = True):
        """
        Değiştirilmiş SVG'yi kaydet ve tarayıcıda aç
        
        Args:
            output_path: Çıktı dosyası yolu
            auto_open: Otomatik olarak tarayıcıda açılsın mı
        """
        # Güzel görünüm için indent ekle
        self._indent(self.root)
        
        # Dosyayı kaydet
        self.tree.write(output_path, encoding='utf-8', xml_declaration=True)
        print(f"✓ Görselleştirilmiş SVG kaydedildi: {output_path}")
        
        # Tarayıcıda aç
        if auto_open:
            abs_path = os.path.abspath(output_path)
            webbrowser.open('file://' + abs_path)
            print(f"✓ SVG tarayıcıda açıldı")
    
    def _indent(self, elem, level=0):
        """XML'i güzel formatlama için indent ekler"""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


def visualize_route(svg_file_path: str,
                   connection_ids: List[str],
                   output_path: Optional[str] = None,
                   route_color: str = "#FF0000",
                   route_width: float = 3.0,
                   anchor_ids: List[str] = None,
                   anchor_color: str = "#FFD700",
                   auto_open: bool = True) -> str:
    """
    Rotayı görselleştirip SVG olarak kaydet ve tarayıcıda aç
    
    Args:
        svg_file_path: Kaynak SVG dosyası
        connection_ids: Rota connection ID'leri
        output_path: Çıktı dosyası (None ise otomatik oluşturulur)
        route_color: Rota rengi (hex)
        route_width: Rota çizgi kalınlığı
        anchor_ids: Dönüş noktalarındaki anchor oda ID'leri
        anchor_color: Anchor rengi (hex)
        auto_open: Otomatik olarak tarayıcıda açılsın mı
    
    Returns:
        Kaydedilen dosyanın yolu
    """
    # Çıktı yolunu belirle
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(svg_file_path))[0]
        output_path = f"route_visualization_{base_name}.svg"
    
    # Visualizer oluştur
    visualizer = RouteVisualizer(svg_file_path)
    
    # Connection'ları highlight et
    count = visualizer.highlight_route_connections(
        connection_ids=connection_ids,
        route_color=route_color,
        route_width=route_width
    )
    
    print(f"  {count} adet connection rengi değiştirildi")
    
    # Anchor odaları highlight et
    if anchor_ids:
        anchor_count = visualizer.highlight_anchor_rooms(
            anchor_ids=anchor_ids,
            anchor_color=anchor_color
        )
        print(f"  {anchor_count} adet anchor oda highlight edildi")
    
    # Kaydet ve aç
    visualizer.save_and_open(output_path, auto_open=auto_open)
    
    return output_path
