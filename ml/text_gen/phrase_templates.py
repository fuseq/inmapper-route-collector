"""
Data-driven phrase template system.

Structure is deterministic (template class selection),
surface form is learned from human annotations (phrase library).

Flow:
  Step → Template Class → Random Variation → Slot Filling → Output
"""
import json
import os
import random
from typing import Dict, List, Optional

# ─────────────────────────────────────────────
#  Turkish suffix tables
# ─────────────────────────────────────────────

_ABLATIVE = {
    '0': 'dan', '1': 'den', '2': 'den', '3': 'ten',
    '4': 'ten', '5': 'ten', '6': 'dan', '7': 'den',
    '8': 'den', '9': 'dan',
    'A': 'dan', 'B': 'den', 'C': 'den', 'D': 'den',
    'E': 'den', 'X': 'ten',
}

_ACCUSATIVE = {
    '0': 'ı',  '1': 'i',  '2': 'yi', '3': 'ü',
    '4': 'ü',  '5': 'i',  '6': 'yı', '7': 'yi',
    '8': 'i',  '9': 'u',
    'A': 'yı', 'B': 'yi', 'C': 'yi', 'D': 'yi',
    'E': 'yi', 'X': 'i',
}

_GENITIVE = {
    '0': 'ın',  '1': 'in',  '2': 'nin', '3': 'ün',
    '4': 'ün',  '5': 'in',  '6': 'nın', '7': 'nin',
    '8': 'in',  '9': 'un',
    'A': 'nın', 'B': 'nin', 'C': 'nin', 'D': 'nin',
    'E': 'nin', 'X': 'in',
}

_DATIVE = {
    '0': 'a',  '1': 'e',  '2': 'ye', '3': 'e',
    '4': 'e',  '5': 'e',  '6': 'ya', '7': 'ye',
    '8': 'e',  '9': 'a',
    'A': 'ya', 'B': 'ye', 'C': 'ye', 'D': 'ye',
    'E': 'ye', 'X': 'e',
}


def _last_char(name: str) -> str:
    stripped = name.rstrip().rstrip(')')
    for ch in reversed(stripped):
        if ch.isalnum():
            return ch.upper()
    return '0'


def _apply_suffix(name: str, table: dict, default: str) -> str:
    last = _last_char(name)
    suffix = table.get(last, default)
    return f"{name}'{suffix}"


def anchor_ablative(name: str) -> str:
    return _apply_suffix(name, _ABLATIVE, 'dan')

def anchor_accusative(name: str) -> str:
    return _apply_suffix(name, _ACCUSATIVE, 'yı')

def anchor_genitive(name: str) -> str:
    return _apply_suffix(name, _GENITIVE, 'in')

def anchor_dative(name: str) -> str:
    return _apply_suffix(name, _DATIVE, 'e')


# ─────────────────────────────────────────────
#  Portal / category mappings
# ─────────────────────────────────────────────

_PORTAL_EXIT_ABL = {
    'Stairs': 'Merdivenden',
    'Elev': 'Asansörden',
    'Escalator': 'Yürüyen merdivenden',
}

_PORTAL_TYPE_TR = {
    'Stairs': 'merdiven',
    'Elev': 'asansör',
    'Escalator': 'yürüyen merdiven',
}

_ROOM_CATEGORY = {
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


# ─────────────────────────────────────────────
#  Load phrase library
# ─────────────────────────────────────────────

_LIB_PATH = os.path.join(os.path.dirname(__file__), 'phrase_library.json')

def _load_library() -> dict:
    with open(_LIB_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data.pop('_meta', None)
    return data

_LIBRARY: Optional[dict] = None

def _get_library() -> dict:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = _load_library()
    return _LIBRARY


# ─────────────────────────────────────────────
#  Template class mapping
# ─────────────────────────────────────────────

def _classify_step(
    step: Dict,
    is_anchor: bool,
    route_context: Optional[Dict] = None,
) -> str:
    """Map a step to its template class name."""
    action = step.get('action', '')

    if action == 'FLOOR_CHANGE':
        return 'FLOOR_CHANGE'

    if action == 'START_PORTAL':
        direction = step.get('direction', '')
        if direction in ('sag', 'saga', 'sağa'):
            return 'START_PORTAL:sag'
        if direction in ('sol', 'sola'):
            return 'START_PORTAL:sol'
        if direction == 'arka':
            return 'START_PORTAL:arka'
        return 'START_PORTAL:duz'

    if action == 'START':
        direction = step.get('direction', '')
        if direction == 'arka':
            return 'START_BEHIND'
        return 'START_SIDE'

    if action == 'ARRIVE':
        if is_anchor:
            return 'ARRIVE_ANCHOR'
        return 'ARRIVE_PROXIMITY'

    if action in ('TURN_LEFT', 'TURN_RIGHT'):
        if is_anchor:
            return 'TURN_ANCHOR'
        return 'NON_ANCHOR_TURN'

    if action == 'VEER':
        if is_anchor:
            return 'VEER_ANCHOR'
        return 'NON_ANCHOR_VEER'

    if action == 'PASS_BY':
        if is_anchor:
            return 'PASS_BY_ANCHOR'
        return 'NON_ANCHOR_PASS_BY'

    return 'NON_ANCHOR_PASS_BY'


def _get_patterns(template_class: str) -> List[str]:
    """Get the pattern list for a template class."""
    lib = _get_library()

    if ':' in template_class:
        base, sub = template_class.split(':', 1)
        entry = lib.get(base, {})
        if isinstance(entry, dict):
            return entry.get(sub, [])
        return []

    entry = lib.get(template_class, [])
    if isinstance(entry, list):
        return entry
    return []


# ─────────────────────────────────────────────
#  Slot filling
# ─────────────────────────────────────────────

def _build_slots(
    step: Dict,
    anchor_name: str,
    route_context: Optional[Dict] = None,
) -> Dict[str, str]:
    """Build a dictionary of all available slot values."""
    action = step.get('action', '') or ''
    direction = step.get('direction', '') or ''

    dir_word = ''
    if action in ('TURN_LEFT',) or direction in ('sol', 'sola'):
        dir_word = 'sola'
    elif action in ('TURN_RIGHT',) or direction in ('sag', 'saga', 'sağa'):
        dir_word = 'sağa'

    side_word = ''
    if direction in ('sol', 'sola'):
        side_word = 'solunuzda'
    elif direction in ('sag', 'saga', 'sağa'):
        side_word = 'sağınızda'
    elif direction == 'arka':
        side_word = 'arkanızda'

    portal_type = step.get('portal_type', '') or ''
    portal_exit = _PORTAL_EXIT_ABL.get(portal_type, 'Portaldan')
    portal_tr = _PORTAL_TYPE_TR.get(portal_type, portal_type)

    end_room = (route_context or {}).get('end_room', {})
    room_type = end_room.get('type', '')
    category = _ROOM_CATEGORY.get(room_type, 'nokta')

    slots = {
        '{anchor}': anchor_name,
        '{anchor_abl}': anchor_ablative(anchor_name) if anchor_name else '',
        '{anchor_acc}': anchor_accusative(anchor_name) if anchor_name else '',
        '{anchor_gen}': anchor_genitive(anchor_name) if anchor_name else '',
        '{anchor_dat}': anchor_dative(anchor_name) if anchor_name else '',
        '{direction}': dir_word,
        '{side}': side_word,
        '{side_raw}': direction,
        '{direction_cap}': dir_word.capitalize() if dir_word else '',
        '{portal}': portal_exit,
        '{portal_tr}': portal_tr,
        '{category}': category,
    }
    return slots


def _fill_pattern(pattern: str, slots: Dict[str, str]) -> str:
    """Replace all slot placeholders in a pattern."""
    result = pattern
    for key, value in slots.items():
        result = result.replace(key, value if value is not None else '')
    return result.strip()


def _filter_valid_patterns(
    patterns: List[str],
    slots: Dict[str, str],
) -> List[str]:
    """Remove patterns that would produce empty slots."""
    valid = []
    for p in patterns:
        filled = _fill_pattern(p, slots)
        if "''" not in filled and "  " not in filled and filled:
            valid.append(p)
    return valid


# ─────────────────────────────────────────────
#  Main API
# ─────────────────────────────────────────────

def generate(
    step: Dict,
    strategy_decision: Optional[Dict] = None,
    anchor_decision: Optional[Dict] = None,
    route_context: Optional[Dict] = None,
    strategy_features: Optional[List[float]] = None,
) -> Optional[str]:
    """
    Generate a description using the phrase library.

    Returns a filled phrase string, or None if no suitable pattern
    is found (caller should fall back to hardcoded templates).
    """
    is_anchor = True
    if strategy_decision:
        is_anchor = strategy_decision.get('anchor_based', True)

    # Resolve anchor name
    anchor_name = _resolve_anchor(step, anchor_decision, route_context)

    template_class = _classify_step(step, is_anchor, route_context)

    if template_class == 'FLOOR_CHANGE':
        return None

    patterns = _get_patterns(template_class)
    if not patterns:
        return None

    slots = _build_slots(step, anchor_name, route_context)
    valid = _filter_valid_patterns(patterns, slots)
    if not valid:
        return None

    chosen = random.choice(valid)
    return _fill_pattern(chosen, slots)


def _resolve_anchor(
    step: Dict,
    anchor_decision: Optional[Dict],
    route_context: Optional[Dict],
) -> str:
    """Get the anchor name from various sources."""
    if anchor_decision and anchor_decision.get('selected_anchor'):
        a = anchor_decision['selected_anchor']
        if isinstance(a, (list, tuple)) and len(a) >= 2:
            return f"{a[0]} - {a[1]}"

    landmark = step.get('landmark', '')
    if landmark:
        return landmark

    action = step.get('action', '')
    if action == 'START':
        start_room = (route_context or {}).get('start_room', {})
        rt = start_room.get('type', '')
        ri = start_room.get('id', '')
        if rt and ri:
            return f"{rt} - {ri}"

    if action == 'ARRIVE':
        end_room = (route_context or {}).get('end_room', {})
        rt = end_room.get('type', '')
        ri = end_room.get('id', '')
        if rt and ri:
            return f"{rt} - {ri}"

    return ''
