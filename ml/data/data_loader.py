"""
Data Loader for ML Training

Loads human-annotated route data from:
  1. Google Sheets (via Apps Script JSON endpoint)
  2. Local submissions/*.json files

Then re-runs route generation to capture full intermediate numeric data
(turn angles, anchor distances, path geometry, visibility scores) that
are not stored in the text-formatted sheet rows.
"""
import json
import os
import re
import sys
import urllib.request
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from helpers.extract_xml import build_graph, get_room_areas
from helpers.path_analysis import NavigationGraphCleaner
from helpers.batch_route_generator import (
    generate_alternative_routes_for_room_pair,
    _find_room_center,
)
from helpers.multi_floor_route_generator import generate_multi_floor_route

# ── Venue configuration (mirrors viewer_app.py) ──

VENUE_CONFIG = {
    'zorlu': {
        'svg_paths': {
            'Kat 1': 'files/floors/1.svg',
            'Kat 0': 'files/floors/0.svg',
            'Kat -1': 'files/floors/-1.svg',
            'Kat -2': 'files/floors/-2.svg',
            'Kat -3': 'files/floors/-3.svg',
        },
        'portals_path': 'files/supportive/portals.json',
        'floor_names': ['Kat 1', 'Kat 0', 'Kat -1', 'Kat -2', 'Kat -3'],
    }
}


class VenueData:
    """Holds loaded venue graphs & floor areas (singleton-style cache)."""

    def __init__(self, venue: str = 'zorlu', base_dir: str = None):
        self.venue = venue
        self.base_dir = base_dir or os.path.join(
            os.path.dirname(__file__), '..', '..'
        )
        self.graphs: List = []
        self.floor_areas: List[Dict] = []
        self.floor_names: List[str] = []
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        cfg = VENUE_CONFIG[self.venue]
        self.floor_names = cfg['floor_names']

        portals_path = os.path.join(self.base_dir, cfg['portals_path'])

        for floor_name in self.floor_names:
            svg_rel = cfg['svg_paths'][floor_name]
            svg_path = os.path.join(self.base_dir, svg_rel)
            graph = build_graph(svg_path, portals_path)
            graph.find_intersections()
            areas = get_room_areas(svg_path)
            self.graphs.append(graph)
            self.floor_areas.append(areas)

        self._loaded = True
        print(f"[VenueData] Loaded {len(self.floor_names)} floors for {self.venue}")

    def find_room(self, floor_name: str, room_id: str) -> Tuple[Optional[Dict], Optional[int], Optional[Dict]]:
        """Return (room_dict, floor_idx, floor_areas) or (None, None, None)."""
        for i, fn in enumerate(self.floor_names):
            if fn == floor_name:
                areas = self.floor_areas[i]
                for room_type, rooms in areas.items():
                    for room in rooms:
                        if room['id'] == room_id:
                            return {
                                'id': room['id'],
                                'type': room_type,
                                'center': room['center'],
                                'area': room['area'],
                            }, i, areas
                return None, None, None
        return None, None, None


# ─────────────────────────────────────────────
#  Parse human annotations
# ─────────────────────────────────────────────

def parse_route_id(route_id: str) -> Dict:
    """
    Parse route_id like 'Kat 0_Food_ID007_to_Kat 0_Shop_ID005'
    into start/end floor + room type + room id.
    """
    m = re.match(
        r'^(Kat\s*-?\d+)_(\w+)_([A-Za-z0-9_-]+)_to_(Kat\s*-?\d+)_(\w+)_([A-Za-z0-9_-]+)$',
        route_id,
    )
    if not m:
        return {}
    return {
        'start_floor': m.group(1),
        'start_type': m.group(2),
        'start_id': m.group(3),
        'end_floor': m.group(4),
        'end_type': m.group(5),
        'end_id': m.group(6),
    }


def parse_human_steps(human_text: str) -> Dict[int, str]:
    """
    Parse numbered human descriptions.
    Returns {step_number: description_text}.
    '-' means DELETE, anything else means KEEP.
    """
    result = {}
    for line in human_text.strip().split('\n'):
        line = line.strip()
        m = re.match(r'^(\d+)\.\s*(.*)', line)
        if m:
            step_num = int(m.group(1))
            desc = m.group(2).strip()
            result[step_num] = desc
    return result


def extract_step_labels(human_steps: Dict[int, str]) -> Dict[int, int]:
    """
    Convert human step descriptions to binary labels.
      '-'        -> 0 (DELETE)
      '+'        -> 1 (KEEP, metric accepted as-is)
      free text  -> 1 (KEEP, human rewrote)
    """
    labels = {}
    for step_num, desc in human_steps.items():
        if desc in ('-', '', '–', '—'):
            labels[step_num] = 0
        else:
            labels[step_num] = 1
    return labels


# ── Regex for detecting room IDs in human text ──
_ROOM_ID_RE = re.compile(r'ID-?\d+[A-Z]*', re.IGNORECASE)

# Turkish portal keywords that count as anchor references
_PORTAL_KEYWORDS_RE = re.compile(
    r'merdiven|asansör|yürüyen\s+merdiven|escalator|elevator',
    re.IGNORECASE,
)

STRATEGY_ANCHOR = 1    # human description references a room/anchor ID
STRATEGY_NO_ANCHOR = 0 # human description uses no room ID (proximity/orientation/geometry)


def _text_references_anchor(desc: str) -> bool:
    """Check if human text references any anchor (room ID or portal keyword)."""
    if _ROOM_ID_RE.search(desc):
        return True
    if _PORTAL_KEYWORDS_RE.search(desc):
        return True
    return False


def extract_strategy_labels(
    human_steps: Dict[int, str],
    step_labels: Dict[int, int],
) -> Dict[int, int]:
    """
    For each KEPT step, determine whether the human used an anchor-based
    description (references a room ID or portal keyword) or a non-anchor
    description.

    Deleted steps (label==0) are excluded from the output.

    Returns {step_num: STRATEGY_ANCHOR or STRATEGY_NO_ANCHOR}.
    """
    labels = {}
    for step_num, desc in human_steps.items():
        if step_labels.get(step_num, 0) == 0:
            continue
        if desc.strip() == '+':
            labels[step_num] = STRATEGY_ANCHOR
        elif _text_references_anchor(desc):
            labels[step_num] = STRATEGY_ANCHOR
        else:
            labels[step_num] = STRATEGY_NO_ANCHOR
    return labels


# Confidence tags returned alongside anchor labels
CONF_CONFIRMED = 'confirmed'   # human typed '+', explicitly accepts primary
CONF_MATCHED   = 'matched'     # human text contains one of the 3 candidate IDs
CONF_UNMATCHED = 'unmatched'   # human wrote text but no candidate ID found
CONF_SKIPPED   = 'skipped'     # step was deleted ('-') or empty


# Map portal type prefixes to Turkish keywords humans use
_PORTAL_TYPE_TO_TR = {
    'stairs':    ['merdiven'],
    'elev':      ['asansör'],
    'elevator':  ['asansör'],
    'escalator': ['yürüyen merdiven'],
}


def _portal_matches_text(anchor_id: str, human_desc: str) -> bool:
    """Check if a portal-type anchor_id matches Turkish keywords in human text."""
    lower_id = anchor_id.lower()
    lower_desc = human_desc.lower()
    for prefix, keywords in _PORTAL_TYPE_TO_TR.items():
        if lower_id.startswith(prefix):
            for kw in keywords:
                if kw in lower_desc:
                    return True
    return False


def extract_anchor_label(
    human_desc: str,
    primary_anchor_id: str,
    alt_anchor_ids: List[str],
) -> Tuple[int, str]:
    """
    Determine which anchor the human referenced (0=primary, 1=alt1, 2=alt2)
    and how confident the label is.

    Portal anchors are matched by Turkish keywords:
      "Stairs.10.-1" <-> "merdiven"
      "Elev.3"       <-> "asansör"
      "Escalator.1"  <-> "yürüyen merdiven"

    Returns (label_index, confidence_tag):
      '-' / empty   -> (0, 'skipped')
      '+'           -> (0, 'confirmed')   human explicitly accepts primary
      text w/ ID    -> (i, 'matched')     human mentioned candidate i
      text w/ portal keyword -> (i, 'matched')  portal type matched
      text w/o ID   -> (0, 'unmatched')   human chose unknown anchor
    """
    if not human_desc or human_desc in ('-', '', '–', '—'):
        return 0, CONF_SKIPPED

    if human_desc.strip() == '+':
        return 0, CONF_CONFIRMED

    all_ids = [primary_anchor_id] + alt_anchor_ids[:2]
    while len(all_ids) < 3:
        all_ids.append(all_ids[-1])

    # First pass: exact ID match
    for i, aid in enumerate(all_ids):
        if aid and aid in human_desc:
            return i, CONF_MATCHED

    # Second pass: portal type keyword match
    for i, aid in enumerate(all_ids):
        if aid and _portal_matches_text(aid, human_desc):
            return i, CONF_MATCHED

    return 0, CONF_UNMATCHED


# ─────────────────────────────────────────────
#  Load from Google Sheets or local files
# ─────────────────────────────────────────────

def load_from_google_sheets(url: str) -> List[Dict]:
    """
    Fetch all rows from the Google Apps Script web app.
    Expects the Apps Script to have a doGet that returns JSON array of rows.
    """
    req = urllib.request.Request(url, method='GET')
    req.add_header('Accept', 'application/json')
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and 'rows' in data:
        return data['rows']
    return []


def _normalize_row_keys(row: Dict) -> Dict:
    """Normalise exported JSON keys (Title Case) to internal snake_case."""
    key_map = {
        'ID': 'id', 'Venue': 'venue',
        'Start Room': 'start_room', 'End Room': 'end_room',
        'Floor': 'floor', 'Metric Steps': 'metric_steps',
        'Human Steps': 'human_steps', 'Timestamp': 'timestamp',
        'Step Count': 'step_count',
    }
    out = {}
    for k, v in row.items():
        out[key_map.get(k, k)] = v
    return out


def load_from_local_submissions(folder: str = 'submissions') -> List[Dict]:
    """Load annotations from the submissions folder.

    Supports:
      - Individual tarif_*.json files (one route per file)
      - A single JSON array export (multiple routes in one file)
    """
    base = os.path.join(os.path.dirname(__file__), '..', '..', folder)
    rows = []
    if not os.path.isdir(base):
        return rows
    for fname in sorted(os.listdir(base)):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(base, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            continue
        if not content.startswith('['):
            content = '[' + content + ']'
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            try:
                data = [json.loads(content.strip('[]\n '))]
            except json.JSONDecodeError:
                continue
        if isinstance(data, list):
            rows.extend(_normalize_row_keys(r) for r in data)
        else:
            rows.append(_normalize_row_keys(data))
    return rows


# ─────────────────────────────────────────────
#  Build route records with full numeric data
# ─────────────────────────────────────────────

def _enrich_turns_with_centers(turns: List[Dict], floor_areas: Dict):
    """Add _anchor_center to each turn for visibility computation."""
    for turn in turns:
        anchor = turn.get('anchor')
        if anchor:
            room_id = anchor[1]
            center = _find_room_center(room_id, floor_areas)
            if center:
                turn['_anchor_center'] = center


def regenerate_route(
    venue_data: VenueData,
    parsed_id: Dict,
    pixel_to_meter: float = 0.1,
) -> Optional[Dict]:
    """
    Re-run route generation for a given parsed route ID.
    Supports both same-floor and multi-floor routes.
    Returns dict with keys: steps, turns, path_points, total_distance, path_connections.
    """
    sf = parsed_id.get('start_floor')
    si = parsed_id.get('start_id')
    ef = parsed_id.get('end_floor')
    ei = parsed_id.get('end_id')

    start_room, start_idx, start_areas = venue_data.find_room(sf, si)
    end_room, end_idx, end_areas = venue_data.find_room(ef, ei)

    if not start_room or not end_room:
        return None

    if sf != ef:
        return _regenerate_multi_floor(
            venue_data, start_room, end_room,
            start_idx, end_idx, sf, ef, pixel_to_meter,
        )

    graph = venue_data.graphs[start_idx]
    alternatives = generate_alternative_routes_for_room_pair(
        start_room=start_room,
        end_room=end_room,
        graph=graph,
        floor_areas=start_areas,
        pixel_to_meter_ratio=pixel_to_meter,
    )

    if not alternatives or 'routes' not in alternatives:
        return None

    route = alternatives['routes'].get('shortest')
    if not route:
        route = list(alternatives['routes'].values())[0]

    turns = route.get('turns', [])
    path_points = route.get('path_points', [])
    steps = route.get('steps', [])
    total_dist = route.get('summary', {}).get('total_distance_meters', 0)

    _enrich_turns_with_centers(turns, start_areas)

    return {
        'steps': steps,
        'turns': turns,
        'path_points': path_points,
        'total_distance': total_dist,
        'path_connections': route.get('path_connections', []),
        'floor': sf,
        'start_room': start_room,
        'end_room': end_room,
    }


def _regenerate_multi_floor(
    venue_data: VenueData,
    start_room: Dict,
    end_room: Dict,
    start_idx: int,
    end_idx: int,
    start_floor: str,
    end_floor: str,
    pixel_to_meter: float,
) -> Optional[Dict]:
    """Call generate_multi_floor_route and normalise to single-floor format."""
    multi_route = generate_multi_floor_route(
        start_room=start_room,
        end_room=end_room,
        start_graph=venue_data.graphs[start_idx],
        end_graph=venue_data.graphs[end_idx],
        all_graphs=venue_data.graphs,
        floor_names=venue_data.floor_names,
        floor_areas_list=venue_data.floor_areas,
        start_floor_name=start_floor,
        end_floor_name=end_floor,
        pixel_to_meter_ratio=pixel_to_meter,
    )

    if not multi_route:
        return None

    turns = multi_route.get('turns', [])

    for segment in multi_route.get('segments', []):
        if segment.get('type') != 'route_segment':
            continue
        seg_floor = segment.get('floor', '')
        for i, fn in enumerate(venue_data.floor_names):
            if fn == seg_floor:
                seg_turns = segment.get('turns', [])
                _enrich_turns_with_centers(seg_turns, venue_data.floor_areas[i])
                break

    return {
        'steps': multi_route['steps'],
        'turns': turns,
        'path_points': multi_route.get('path_points', []),
        'total_distance': multi_route['summary']['total_distance_meters'],
        'path_connections': [],
        'floor': f"{start_floor}->{end_floor}",
        'start_room': start_room,
        'end_room': end_room,
        'is_multi_floor': True,
        'segments': multi_route.get('segments', []),
    }


# ─────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────

def build_training_dataset(
    source: str = 'local',
    sheets_url: str = '',
    venue: str = 'zorlu',
    base_dir: str = None,
) -> List[Dict]:
    """
    Build the full training dataset by:
      1. Loading human annotations
      2. Re-running route generation for numeric features
      3. Aligning labels with regenerated steps
    
    Returns list of route_records, each containing:
      - steps, turns, path_points, total_distance  (for FeatureExtractor)
      - step_labels: {step_num: 0 or 1}            (for StepFilter)
      - anchor_labels: {step_num: 0/1/2}           (for AnchorSelection)
      - anchor_confidence: {step_num: str}          (confidence tag per anchor label)
      - strategy_labels: {step_num: 0 or 1}         (for DescriptionStrategy, kept steps only)
      - route_id: str
    """
    if source == 'sheets' and sheets_url:
        print("[DataLoader] Fetching from Google Sheets...")
        raw_rows = load_from_google_sheets(sheets_url)
    else:
        print("[DataLoader] Loading from local submissions...")
        raw_rows = load_from_local_submissions()

    print(f"[DataLoader] {len(raw_rows)} raw annotation rows loaded")

    venue_data = VenueData(venue=venue, base_dir=base_dir)
    venue_data.load()

    dataset = []
    skipped = 0

    for row in raw_rows:
        route_id = row.get('id', '')
        parsed = parse_route_id(route_id)
        if not parsed:
            skipped += 1
            continue

        route_data = regenerate_route(venue_data, parsed)
        if not route_data:
            skipped += 1
            continue

        human_text = row.get('human_steps', '')
        human_steps = parse_human_steps(human_text)
        step_labels = extract_step_labels(human_steps)

        anchor_labels = {}
        anchor_confidence = {}
        for step in route_data['steps']:
            sn = step.get('step_number', 0)
            lm = step.get('landmark', '')
            alt_lms = step.get('alt_landmarks', [])
            primary_id = ''
            if lm:
                parts = lm.split(' - ', 1)
                if len(parts) == 2:
                    primary_id = parts[1].strip()
            alt_ids = []
            for alm in (alt_lms or []):
                parts = alm.split(' - ', 1)
                if len(parts) == 2:
                    alt_ids.append(parts[1].strip())

            h_desc = human_steps.get(sn, '')
            label, conf = extract_anchor_label(h_desc, primary_id, alt_ids)
            anchor_labels[sn] = label
            anchor_confidence[sn] = conf

        strategy_labels = extract_strategy_labels(human_steps, step_labels)

        route_data['step_labels'] = step_labels
        route_data['anchor_labels'] = anchor_labels
        route_data['anchor_confidence'] = anchor_confidence
        route_data['strategy_labels'] = strategy_labels
        route_data['route_id'] = route_id
        route_data['_raw_human_text'] = human_text

        dataset.append(route_data)

    print(f"[DataLoader] Built {len(dataset)} route records ({skipped} skipped)")
    return dataset
