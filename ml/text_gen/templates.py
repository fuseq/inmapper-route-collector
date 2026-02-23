"""
Template-Based Description Generator

Two-tier system:
  1. Phrase library (data-driven): patterns extracted from human annotations
     with slot-based variation. Provides natural, diverse phrasing.
  2. Hardcoded fallback: deterministic templates for cases where the
     phrase library has no suitable pattern.
"""
import random
from typing import Dict, List, Optional

from ml.text_gen.phrase_templates import generate as phrase_generate


def generate_description(
    step: Dict,
    strategy_decision: Optional[Dict] = None,
    anchor_decision: Optional[Dict] = None,
    route_context: Optional[Dict] = None,
    strategy_features: Optional[List[float]] = None,
) -> str:
    """
    Generate a template-based description for a single step.

    Tries the phrase library first; falls back to hardcoded templates.
    """
    action = step.get('action', 'START')

    if action == 'FLOOR_CHANGE':
        return _floor_change_description(step)

    # Phrase library (data-driven, first priority)
    result = phrase_generate(
        step, strategy_decision, anchor_decision,
        route_context, strategy_features,
    )
    if result:
        return result

    # Hardcoded fallback below
    is_anchor = True
    if strategy_decision:
        is_anchor = strategy_decision.get('anchor_based', True)

    if not is_anchor:
        return _non_anchor_description(step, route_context, strategy_features)

    if action == 'START':
        return _anchor_start(step, anchor_decision, route_context)
    elif action == 'START_PORTAL':
        return _start_portal_description(step)
    elif action == 'ARRIVE':
        return _anchor_arrive(step, anchor_decision, route_context)
    elif action in ('TURN_LEFT', 'TURN_RIGHT'):
        return _anchor_turn(step, anchor_decision)
    elif action == 'VEER':
        return _anchor_veer(step, anchor_decision)
    elif action == 'PASS_BY':
        return _anchor_pass_by(step, anchor_decision)
    else:
        return _anchor_generic(step, anchor_decision)


# ─────────────────────────────────────────────
#  Floor change template
# ─────────────────────────────────────────────

_PORTAL_TYPE_TR = {
    'Stairs': 'Merdiven',
    'Elev': 'Asansör',
    'Escalator': 'Yürüyen merdiven',
}


def _floor_change_description(step: Dict) -> str:
    portal_type = step.get('portal_type', '')
    to_floor = step.get('to_floor', '')
    tr_type = _PORTAL_TYPE_TR.get(portal_type, portal_type)
    if to_floor:
        return f"{tr_type} ile {to_floor} katına geçin"
    return step.get('description', f"{tr_type} ile kata geçin")


_PORTAL_EXIT_TR = {
    'Stairs': 'Merdivenden',
    'Elev': 'Asansörden',
    'Escalator': 'Yürüyen merdivenden',
}


def _start_portal_description(step: Dict) -> str:
    portal_type = step.get('portal_type', '')
    tr_exit = _PORTAL_EXIT_TR.get(portal_type, portal_type)
    if not tr_exit:
        tr_exit = 'Portaldan'

    direction = step.get('direction', '')

    if direction == 'sag':
        return f"{tr_exit} çıkınca sağa doğru ilerleyin"
    if direction == 'sol':
        return f"{tr_exit} çıkınca sola doğru ilerleyin"
    if direction == 'arka':
        return f"{tr_exit} çıkınca arkaya dönüp ilerleyin"
    return f"{tr_exit} çıkıp düz ilerleyin"


# ─────────────────────────────────────────────
#  Anchor-based templates
# ─────────────────────────────────────────────

def _get_anchor_label(step: Dict, anchor_decision: Optional[Dict]) -> str:
    if anchor_decision and anchor_decision.get('selected_anchor'):
        a = anchor_decision['selected_anchor']
        if isinstance(a, (list, tuple)) and len(a) >= 2:
            return f"{a[0]} - {a[1]}"
    return step.get('landmark', '')


def _format_distance(meters: float) -> str:
    """Human-friendly distance without exact meters."""
    if meters <= 0:
        return ''
    if meters < 5:
        return 'biraz'
    if meters < 15:
        return 'bir süre'
    return 'yolun sonuna kadar'


def _side_text(direction: str) -> str:
    mapping = {
        'sol': 'solunuzda', 'sola': 'solunuzda',
        'sag': 'sağınızda', 'saga': 'sağınızda', 'sağa': 'sağınızda',
        'arka': 'arkanızda',
    }
    return mapping.get(direction, '')


def _direction_text(action: str) -> str:
    if action == 'TURN_LEFT':
        return 'sola'
    if action == 'TURN_RIGHT':
        return 'sağa'
    return ''


_TURN_VERBS = ['dönün', 'yönelin', 'gidin']
_MOVE_VERBS = ['ilerleyin', 'devam edin', 'yürüyün', 'gidin']
_CONNECTORS = [' ve ', ' sonra ', ' ardından ', ', ']


def _anchor_start(
    step: Dict,
    anchor_decision: Optional[Dict],
    route_context: Optional[Dict],
) -> str:
    anchor = _get_anchor_label(step, anchor_decision)
    if not anchor:
        start_room = (route_context or {}).get('start_room', {})
        room_type = start_room.get('type', '')
        room_id = start_room.get('id', '')
        if room_type and room_id:
            anchor = f"{room_type} - {room_id}"

    direction = step.get('direction', '')
    side = _side_text(direction)

    if direction == 'arka':
        if anchor:
            return f"{anchor} arkanızda kalacak şekilde ilerleyin"
        return "Başlangıç noktasından çıkıp düz ilerleyin"

    if anchor and side:
        return f"{anchor} {side} kalacak şekilde ilerleyin"
    if anchor:
        return f"{anchor} noktasından ilerleyin"
    return "Düz ilerleyin"


def _anchor_arrive(
    step: Dict,
    anchor_decision: Optional[Dict],
    route_context: Optional[Dict],
) -> str:
    end_room = (route_context or {}).get('end_room', {})
    room_id = end_room.get('id', '')
    room_type = end_room.get('type', '')

    anchor = _get_anchor_label(step, anchor_decision)
    if anchor:
        return f"{anchor} konumuna ulaştınız"

    if room_type and room_id:
        return f"{room_type} - {room_id} konumuna ulaştınız"
    return "Hedefe ulaştınız"


def _anchor_turn(step: Dict, anchor_decision: Optional[Dict]) -> str:
    anchor = _get_anchor_label(step, anchor_decision)
    action = step.get('action', '')
    direction = _direction_text(action)
    verb = random.choice(_TURN_VERBS)
    move = random.choice(_MOVE_VERBS)

    if anchor:
        return f"{anchor}'den {direction} dönüp {move}"
    return f"{direction.capitalize()} {verb} ve {move}"


def _anchor_veer(step: Dict, anchor_decision: Optional[Dict]) -> str:
    anchor = _get_anchor_label(step, anchor_decision)
    direction = step.get('direction', '')
    side = _side_text(direction)
    move = random.choice(_MOVE_VERBS)

    dir_word = 'sola' if direction in ('sol', 'sola') else 'sağa'

    if anchor and side:
        return f"{anchor} {side} kalacak şekilde {dir_word} doğru {move}"
    if anchor:
        return f"{anchor}'den {dir_word} doğru {move}"
    return f"Hafif {dir_word} kayarak {move}"


def _anchor_pass_by(step: Dict, anchor_decision: Optional[Dict]) -> str:
    anchor = _get_anchor_label(step, anchor_decision)
    direction = step.get('direction', '')
    side = _side_text(direction)
    move = random.choice(_MOVE_VERBS)

    if anchor and side:
        return f"{anchor} {side} kalacak şekilde {move}"
    if anchor:
        return f"{anchor} yanından geçerek {move}"
    return f"Düz {move}"


def _anchor_generic(step: Dict, anchor_decision: Optional[Dict]) -> str:
    move = random.choice(_MOVE_VERBS)
    return f"Düz {move}"


# ─────────────────────────────────────────────
#  Non-anchor templates
# ─────────────────────────────────────────────

def _non_anchor_description(
    step: Dict,
    route_context: Optional[Dict],
    strategy_features: Optional[List[float]],
) -> str:
    action = step.get('action', '')
    room_behind = False
    rooms_adjacent = False
    junction_high = False

    if strategy_features and len(strategy_features) >= 16:
        room_behind = strategy_features[10] > 0.5
        junction_high = strategy_features[14] > 0.3
        rooms_adjacent = strategy_features[15] > 0.5

    if action == 'START':
        return _non_anchor_start(step, room_behind, rooms_adjacent)
    elif action == 'ARRIVE':
        return _non_anchor_arrive(
            step, route_context, rooms_adjacent, junction_high,
        )
    else:
        return _non_anchor_mid(step)


def _non_anchor_start(
    step: Dict,
    room_behind: bool,
    rooms_adjacent: bool,
) -> str:
    if room_behind:
        return "Burayı arkanıza alıp düz ilerleyin"
    if rooms_adjacent:
        return "Düz ilerleyin, hedefiniz çok yakın"
    return "Düz ilerleyin"


def _non_anchor_arrive(
    step: Dict,
    route_context: Optional[Dict],
    rooms_adjacent: bool,
    junction_high: bool,
) -> str:
    end_room = (route_context or {}).get('end_room', {})
    room_type = end_room.get('type', '')

    type_label = _room_type_label(room_type)

    if rooms_adjacent:
        return f"Hemen yandaki {type_label}"
    if junction_high:
        return f"Köşedeki {type_label}"

    return f"Karşınızdaki {type_label}"


def _non_anchor_mid(step: Dict) -> str:
    """Non-anchor description for middle steps (rare case)."""
    action = step.get('action', '')
    direction = _direction_text(action)
    move = random.choice(_MOVE_VERBS)
    if direction:
        return f"{direction.capitalize()} dönüp {move}"
    return f"Düz {move}"


def _room_type_label(room_type: str) -> str:
    """Convert room type code to a natural Turkish label."""
    mapping = {
        'Food': 'restoran',
        'Shop': 'mağaza',
        'Medical': 'sağlık birimi',
        'Social': 'alan',
        'Other': 'nokta',
        'Elevator': 'asansör',
        'Stairs': 'merdiven',
        'Escalator': 'yürüyen merdiven',
        'WC': 'tuvalet',
        'Parking': 'otopark',
        'Info': 'bilgi noktası',
    }
    return mapping.get(room_type, 'nokta')
