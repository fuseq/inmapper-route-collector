"""
Text Generation Formatter — Placeholder Approach

Converts structured step data into T5 input text with [ANCHOR] placeholders
instead of real anchor names. This forces the model to learn sentence structure
and Turkish grammar rather than memorising specific anchor IDs.

T5 input format (new):
    describe: | action=TURN_RIGHT | strategy=anchor | anchor=[ANCHOR]
              | anchor_type=Shop | suffix=den | side=sag | is_first=0 | is_last=0

Target (new):
    [ANCHOR]'den sağa dönüp ilerleyin

At inference time, [ANCHOR] is replaced with the real anchor name and the
suffix is corrected for Turkish vowel harmony.
"""
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

PIXEL_TO_METER = 0.1

PLACEHOLDER = '[ANCHOR]'

# ─────────────────────────────────────────────
#  Turkish suffix helper
# ─────────────────────────────────────────────

_ABLATIVE_BY_LAST = {
    '0': 'dan', '1': 'den', '2': 'den', '3': 'ten',
    '4': 'ten', '5': 'ten', '6': 'dan', '7': 'den',
    '8': 'den', '9': 'dan',
}

_ACCUSATIVE_BY_LAST = {
    '0': 'ı',  '1': 'i',  '2': 'yi', '3': 'ü',
    '4': 'ü',  '5': 'i',  '6': 'yı', '7': 'yi',
    '8': 'i',  '9': 'u',
}


def get_ablative_suffix(anchor_name: str) -> str:
    """Return the Turkish ablative suffix (-den/-dan/-ten/-tan) for an anchor."""
    last = _last_significant_char(anchor_name)
    return _ABLATIVE_BY_LAST.get(last, 'dan')


def get_accusative_suffix(anchor_name: str) -> str:
    """Return the Turkish accusative suffix (-i/-ı/-u/-ü/-yi/...) for an anchor."""
    last = _last_significant_char(anchor_name)
    return _ACCUSATIVE_BY_LAST.get(last, 'yı')


def _last_significant_char(name: str) -> str:
    """Get the last alphanumeric character (upper-cased) for suffix lookup."""
    stripped = name.rstrip().rstrip(')')
    for ch in reversed(stripped):
        if ch.isalnum():
            return ch.upper()
    return '0'


def _anchor_type_from_id(anchor_name: str) -> str:
    """Extract the type prefix (Shop, Food, Stairs, Elev, etc.) from an anchor."""
    if not anchor_name:
        return ''
    parts = anchor_name.split(' - ', 1)
    return parts[0].strip() if parts else ''


def _is_portal_anchor(anchor_type: str) -> bool:
    return anchor_type.lower() in ('stairs', 'elev', 'elevator', 'escalator')


_PORTAL_TR = {
    'stairs': 'merdiven',
    'elev': 'asansör',
    'elevator': 'asansör',
    'escalator': 'yürüyen merdiven',
}


def _portal_turkish(anchor_type: str) -> str:
    return _PORTAL_TR.get(anchor_type.lower(), anchor_type)


# ─────────────────────────────────────────────
#  Format T5 input string
# ─────────────────────────────────────────────

def format_step_input(
    step: Dict,
    strategy_decision: Optional[Dict] = None,
    anchor_decision: Optional[Dict] = None,
    route_context: Optional[Dict] = None,
    strategy_features: Optional[List[float]] = None,
    prev_action: str = '',
) -> str:
    """
    Build a T5 input string for a single step.

    The anchor name is replaced with [ANCHOR] and additional hints
    (anchor_type, suffix, after_floor_change) are added so the model
    can learn Turkish grammar without memorising specific IDs.
    """
    parts = ['describe:']

    action = step.get('action', 'START')
    sn = step.get('step_number', 0)

    parts.append(f'action={action}')

    is_anchor = True
    if strategy_decision:
        is_anchor = strategy_decision.get('anchor_based', True)
    parts.append(f'strategy={"anchor" if is_anchor else "non_anchor"}')

    anchor_name = ''
    anchor_type = ''

    if is_anchor:
        anchor_name = _get_anchor_name(step, anchor_decision)
        anchor_type = _anchor_type_from_id(anchor_name)

        if _is_portal_anchor(anchor_type):
            parts.append(f'anchor_type={anchor_type}')
        elif anchor_name:
            parts.append(f'anchor={PLACEHOLDER}')
            parts.append(f'anchor_type={anchor_type}')
            suffix = get_ablative_suffix(anchor_name)
            parts.append(f'suffix={suffix}')

        direction = step.get('direction', '')
        if direction:
            parts.append(f'side={direction}')

    is_first = sn == 1
    is_last = action == 'ARRIVE'
    parts.append(f'is_first={int(is_first)}')
    parts.append(f'is_last={int(is_last)}')

    if prev_action == 'FLOOR_CHANGE':
        portal_type = step.get('portal_type', '')
        parts.append('after_floor_change=1')
        if portal_type:
            parts.append(f'portal_type={portal_type}')

    if route_context:
        total_dist = route_context.get('total_distance', 0)
        if total_dist > 0:
            parts.append(f'total_dist={total_dist:.0f}')

    if strategy_features and len(strategy_features) >= 16:
        room_behind = strategy_features[10]
        rooms_adjacent = strategy_features[15]
        junction_deg = strategy_features[14]
        if room_behind > 0.5:
            parts.append('room_behind=1')
        if rooms_adjacent > 0.5:
            parts.append('rooms_adjacent=1')
        if junction_deg > 0.3:
            parts.append(f'junction={junction_deg:.1f}')

    return ' | '.join(parts)


def _get_anchor_name(step: Dict, anchor_decision: Optional[Dict]) -> str:
    """Extract the anchor name to use."""
    if anchor_decision and anchor_decision.get('selected_anchor'):
        anchor = anchor_decision['selected_anchor']
        if isinstance(anchor, (list, tuple)) and len(anchor) >= 2:
            return f"{anchor[0]} - {anchor[1]}"

    landmark = step.get('landmark', '')
    if landmark:
        return landmark
    return ''


# ─────────────────────────────────────────────
#  Build training pairs
# ─────────────────────────────────────────────

_ROOM_ID_RE = re.compile(r'((?:Shop|Food|Other|Medical|Social|Stairs|Elevator|Elev|Escalator)\s*-\s*[A-Za-z0-9._-]+)', re.IGNORECASE)


def _replace_anchor_in_target(target: str, anchor_name: str) -> str:
    """
    Replace the anchor name reference in the human-written target text
    with [ANCHOR]. Handles both exact matches and partial ID matches.
    """
    if not anchor_name or not target:
        return target

    if anchor_name in target:
        return target.replace(anchor_name, PLACEHOLDER, 1)

    parts = anchor_name.split(' - ', 1)
    if len(parts) == 2:
        room_id = parts[1].strip()
        if room_id in target:
            match = _ROOM_ID_RE.search(target)
            if match and room_id in match.group(0):
                return target.replace(match.group(0), PLACEHOLDER, 1)

    return target


def format_training_pair(
    route_record: Dict,
    step_num: int,
    human_desc: str,
) -> Optional[Tuple[str, str]]:
    """
    Build a (input_text, target_text) training pair with placeholder anchors.
    """
    if not human_desc or human_desc in ('-', '', '–', '—'):
        return None

    steps = route_record.get('steps', [])
    step = None
    step_idx = -1
    for i, s in enumerate(steps):
        if s.get('step_number', 0) == step_num:
            step = s
            step_idx = i
            break

    if step is None:
        return None

    strategy_labels = route_record.get('strategy_labels', {})
    is_anchor = strategy_labels.get(step_num, 1) == 1
    strategy_decision = {'anchor_based': is_anchor, 'anchor_prob': 1.0}

    anchor_decision = None
    anchor_name = ''
    anchor_labels = route_record.get('anchor_labels', {})
    if is_anchor and step.get('landmark'):
        selected_idx = anchor_labels.get(step_num, 0)
        candidates = []
        lm = step.get('landmark', '')
        if lm:
            parts = lm.split(' - ', 1)
            if len(parts) == 2:
                candidates.append((parts[0].strip(), parts[1].strip()))
        for alm in (step.get('alt_landmarks') or []):
            parts = alm.split(' - ', 1)
            if len(parts) == 2:
                candidates.append((parts[0].strip(), parts[1].strip()))
        if candidates:
            sel_idx = min(selected_idx, len(candidates) - 1)
            anchor_decision = {
                'selected_anchor': candidates[sel_idx],
                'candidates': candidates,
            }
            anchor_name = f"{candidates[sel_idx][0]} - {candidates[sel_idx][1]}"

    # Determine prev_action for after_floor_change detection
    prev_action = ''
    if step_idx > 0:
        prev_step = steps[step_idx - 1]
        prev_action = prev_step.get('action', '')

    # Build target text
    anchor_type = _anchor_type_from_id(anchor_name)

    if human_desc.strip() == '+':
        from ml.text_gen.templates import generate_description as template_gen
        route_context = {
            'total_distance': route_record.get('total_distance', 0),
            'start_room': route_record.get('start_room', {}),
            'end_room': route_record.get('end_room', {}),
        }
        sf = _extract_spatial_hints(route_record, step_idx)
        target = template_gen(step, strategy_decision, anchor_decision, route_context, sf)
        if is_anchor and anchor_name and not _is_portal_anchor(anchor_type):
            target = _replace_anchor_in_target(target, anchor_name)
    else:
        target = human_desc
        if is_anchor and anchor_name and not _is_portal_anchor(anchor_type):
            target = _replace_anchor_in_target(target, anchor_name)

    route_context = {
        'total_distance': route_record.get('total_distance', 0),
        'start_room': route_record.get('start_room', {}),
        'end_room': route_record.get('end_room', {}),
    }

    spatial_features = _extract_spatial_hints(route_record, step_idx)

    input_text = format_step_input(
        step=step,
        strategy_decision=strategy_decision,
        anchor_decision=anchor_decision,
        route_context=route_context,
        strategy_features=spatial_features,
        prev_action=prev_action,
    )

    return input_text, target


def _extract_spatial_hints(route_record: Dict, step_idx: int) -> List[float]:
    """
    Build a minimal 16-element list with spatial hints for the formatter.
    Only indices 10 (room_behind), 14 (junction_degree), and 15 (rooms_adjacent)
    are used by format_step_input.
    """
    features = [0.0] * 16

    step = route_record['steps'][step_idx] if step_idx >= 0 else {}
    action = step.get('action', '')
    direction = step.get('direction', '')

    if action == 'START' and direction == 'arka':
        features[10] = 1.0

    start_room = route_record.get('start_room', {})
    end_room = route_record.get('end_room', {})
    sc = start_room.get('center')
    ec = end_room.get('center')
    if sc and ec:
        dx = ec[0] - sc[0]
        dy = ec[1] - sc[1]
        direct_m = math.sqrt(dx * dx + dy * dy) * PIXEL_TO_METER
        if direct_m < 10.0:
            features[15] = 1.0

    return features


def build_all_training_pairs(route_records: List[Dict]) -> List[Tuple[str, str]]:
    """
    Iterate all route records and produce training pairs for every kept step
    that has a human description.
    """
    from ml.data.data_loader import parse_human_steps, extract_step_labels

    pairs = []
    for rec in route_records:
        human_text = rec.get('_raw_human_text', '')
        if not human_text:
            continue

        human_steps = parse_human_steps(human_text)
        step_labels = extract_step_labels(human_steps)

        for sn, desc in human_steps.items():
            if step_labels.get(sn, 0) == 0:
                continue
            pair = format_training_pair(rec, sn, desc)
            if pair:
                pairs.append(pair)

    return pairs
