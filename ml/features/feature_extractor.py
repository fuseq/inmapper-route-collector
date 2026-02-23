"""
Feature Extractor for Indoor Navigation ML Models

Converts raw route generation output (turns, steps, path_points) into
fixed-size numeric feature vectors suitable for PyTorch models.

Three feature sets are produced:
  - Step features (20-dim) for the StepFilter model
  - Anchor candidate features (24-dim each) for the AnchorSelection model
    organised as three layers:
      A) Per-candidate metrics  (11 numeric + 1 type_idx)
      B) Turn context           (6 numeric, shared across candidates)
      C) Comparative features   (6 numeric, relative to other candidates)
  - Strategy features (16-dim) for the DescriptionStrategy model
    Predicts whether a kept step uses an anchor-based or non-anchor description.
"""
import math
import sys
import os
from typing import List, Tuple, Dict, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from helpers.path_analysis import distance
from helpers.anchor_selector import (
    is_in_view_cone,
    get_anchor_tier,
    get_distance_score,
    get_size_score,
)

# ── Action type encoding ──
ACTION_TYPES = ['START', 'TURN_LEFT', 'TURN_RIGHT', 'VEER', 'PASS_BY', 'FLOOR_CHANGE', 'ARRIVE']
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_TYPES)}

# ── Room type encoding (for anchor embedding) ──
ROOM_TYPES = [
    'Food', 'Shop', 'Medical', 'Social', 'Other',
    'Elevator', 'Stairs', 'Parking', 'Portal', 'Commercial',
    'Escalator', 'WC', 'Info', 'Unknown',
]
ROOM_TYPE_TO_IDX = {t: i for i, t in enumerate(ROOM_TYPES)}
NUM_ROOM_TYPES = len(ROOM_TYPES)

# ── Turn type encoding for context features ──
TURN_TYPES = ['turn', 'bend', 'veer']

# ── Normalization constants ──
MAX_DISTANCE_M = 200.0
MAX_ANCHOR_DIST_M = 50.0
PIXEL_TO_METER = 0.1

# Position of the type_idx column in the raw feature vector (last position)
ANCHOR_TYPE_IDX_COL = 23
# Total raw feature dimension per candidate (23 numeric + 1 type_idx)
ANCHOR_RAW_DIM = 24
# Numeric dimension after removing type_idx column
ANCHOR_NUMERIC_DIM = 23

# Strategy features dimension
STRATEGY_FEATURE_DIM = 16

# Adjacency threshold: rooms closer than this (metres) are "adjacent"
ADJACENT_THRESHOLD_M = 10.0


def _room_type_index(room_type: str) -> int:
    """Map a room type string to its integer index."""
    for key in ROOM_TYPE_TO_IDX:
        if key.lower() in room_type.lower():
            return ROOM_TYPE_TO_IDX[key]
    return ROOM_TYPE_TO_IDX['Unknown']


def _compute_relative_anchor_position(
    turn_point: Tuple[float, float],
    anchor_center: Tuple[float, float],
    incoming_vector: Tuple[float, float],
) -> float:
    """
    Determine whether the anchor is before (-1), at (0), or after (+1) 
    the turn point, relative to the walking direction.
    """
    if incoming_vector is None:
        return 0.0

    in_mag = math.hypot(*incoming_vector)
    if in_mag < 1e-6:
        return 0.0

    in_norm = (incoming_vector[0] / in_mag, incoming_vector[1] / in_mag)
    dx = anchor_center[0] - turn_point[0]
    dy = anchor_center[1] - turn_point[1]

    dot = in_norm[0] * dx + in_norm[1] * dy
    dist = math.hypot(dx, dy)
    if dist < 5.0:
        return 0.0
    cos_val = dot / dist
    if cos_val > 0.3:
        return 1.0   # ahead / after
    elif cos_val < -0.3:
        return -1.0   # behind / before
    return 0.0


def _angle_alignment(
    turn_point: Tuple[float, float],
    anchor_center: Tuple[float, float],
    outgoing_vector: Tuple[float, float],
) -> float:
    """Cosine similarity between turn outgoing direction and anchor direction."""
    if outgoing_vector is None:
        return 0.0

    out_mag = math.hypot(*outgoing_vector)
    if out_mag < 1e-6:
        return 0.0

    out_norm = (outgoing_vector[0] / out_mag, outgoing_vector[1] / out_mag)
    dx = anchor_center[0] - turn_point[0]
    dy = anchor_center[1] - turn_point[1]
    d_mag = math.hypot(dx, dy)
    if d_mag < 1e-6:
        return 1.0

    return (out_norm[0] * dx + out_norm[1] * dy) / d_mag


class FeatureExtractor:
    """
    Extracts numeric feature vectors from route generation output.
    
    Expects a 'route_record' dict with:
        - turns: list of turn dicts from _detect_turns_from_path
        - steps: list of RouteStep dicts (to_dict() format)
        - path_points: list of (x, y) tuples
        - total_distance: float (metres)
    
    Step features:   20-dim per step
    Anchor features: 24-dim per candidate (23 numeric + 1 type_idx)
        A) Per-candidate metrics  [0..10]  : 11 numeric
        B) Turn context           [11..16] :  6 numeric
        C) Comparative            [17..22] :  6 numeric
        type_idx                  [23]     :  1 int (embedding)
    """

    def __init__(self, pixel_to_meter: float = PIXEL_TO_METER):
        self.px2m = pixel_to_meter

    # ─────────────────────────────────────────────
    #  STEP FILTER FEATURES  (20-dim per step)
    # ─────────────────────────────────────────────

    def extract_step_features(self, route_record: Dict) -> np.ndarray:
        """
        Return (N_steps, 20) float32 array of step-level features.
        """
        steps = route_record['steps']
        turns = route_record.get('turns', [])
        path_points = route_record.get('path_points', [])
        total_dist = route_record.get('total_distance', 1.0) or 1.0

        turn_by_index = self._index_turns(turns, steps, path_points)

        features = []
        for step in steps:
            features.append(
                self._step_feature_vector(step, turn_by_index, total_dist, steps)
            )
        return np.array(features, dtype=np.float32)

    def _step_feature_vector(
        self,
        step: Dict,
        turn_by_step: Dict[int, Dict],
        total_distance: float,
        steps: List[Dict],
    ) -> List[float]:
        action = step.get('action', 'START')
        step_num = step.get('step_number', 0)
        turn = turn_by_step.get(step_num)

        # 1) action one-hot (7 dim)
        action_oh = [0.0] * len(ACTION_TYPES)
        idx = ACTION_TO_IDX.get(action, 0)
        action_oh[idx] = 1.0

        # 2) turn_angle normalised [0, 1]
        turn_angle = (turn['angle'] / 180.0) if turn else 0.0

        # 3) path_length normalised
        path_length = step.get('distance_meters', 0.0) / MAX_DISTANCE_M
        path_length = min(path_length, 1.0)

        # 4) cumulative path ratio
        cum_dist = step.get('cumulative_distance', 0.0)
        cum_ratio = cum_dist / total_distance if total_distance > 0 else 0.0
        cum_ratio = min(cum_ratio, 1.0)

        # 5) anchor_distance_to_turn normalised
        anchor_dist = 0.0
        if turn and turn.get('anchor'):
            raw_dist = turn['anchor'][3] * self.px2m
            anchor_dist = min(raw_dist / MAX_ANCHOR_DIST_M, 1.0)

        # 6) anchor_visibility_score
        vis_score = 0.0
        if turn and turn.get('_visibility_score') is not None:
            vis_score = turn['_visibility_score']

        # 7) same_side_flag
        same_side = 0.0
        if turn:
            anchor_side = turn.get('anchor_side')
            direction = turn.get('direction', '')
            if anchor_side and direction:
                side_map = {'sol': 'sola', 'sag': 'sağa'}
                if side_map.get(anchor_side) == direction:
                    same_side = 1.0

        # 8) relative_anchor_position (-1, 0, +1)
        rel_pos = 0.0
        if turn and turn.get('_relative_anchor_position') is not None:
            rel_pos = turn['_relative_anchor_position']

        # 9) zigzag_indicator
        zigzag = 1.0 if (turn and turn.get('turn_type') == 'veer') else 0.0

        # 10-14) context features for portal vs junction disambiguation
        portal_prox = self._portal_proximity(step_num, steps)
        seg_turn_cnt = self._segment_turn_count(step_num, steps)
        only_turn = self._is_only_turn_in_segment(step_num, steps)
        dist_prev = self._dist_to_adjacent_turn(step_num, steps, -1)
        dist_next = self._dist_to_adjacent_turn(step_num, steps, +1)

        return action_oh + [
            turn_angle, path_length, cum_ratio,
            anchor_dist, vis_score, same_side, rel_pos, zigzag,
            portal_prox, seg_turn_cnt, only_turn, dist_prev, dist_next,
        ]

    # ── Step filter context helpers ──

    @staticmethod
    def _portal_proximity(step_num: int, steps: List[Dict]) -> float:
        """
        Distance (in step count) to nearest FLOOR_CHANGE step.
        Returns 1.0 when far (>=5 steps), 0.0 when adjacent.
        """
        min_dist = 999
        for s in steps:
            if s.get('action') == 'FLOOR_CHANGE':
                d = abs(s.get('step_number', 0) - step_num)
                if d < min_dist:
                    min_dist = d
        if min_dist == 999:
            return 1.0
        return min(min_dist, 5) / 5.0

    @staticmethod
    def _segment_turn_count(step_num: int, steps: List[Dict]) -> float:
        """
        Count TURN_LEFT/TURN_RIGHT steps in the same segment
        (bounded by START, FLOOR_CHANGE, or ARRIVE).
        Normalised by dividing by 6.
        """
        boundaries = {'START', 'FLOOR_CHANGE', 'ARRIVE'}
        seg_start = 0
        seg_end = len(steps)
        for i, s in enumerate(steps):
            sn = s.get('step_number', 0)
            act = s.get('action', '')
            if act in boundaries and sn < step_num:
                seg_start = i
            if act in boundaries and sn > step_num and i < seg_end:
                seg_end = i
                break

        count = 0
        for s in steps[seg_start:seg_end + 1]:
            if s.get('action') in ('TURN_LEFT', 'TURN_RIGHT'):
                count += 1
        return min(count, 6) / 6.0

    @staticmethod
    def _is_only_turn_in_segment(step_num: int, steps: List[Dict]) -> float:
        """1.0 if this is the only TURN_LEFT/TURN_RIGHT in its segment."""
        cur_step = None
        for s in steps:
            if s.get('step_number', 0) == step_num:
                cur_step = s
                break
        if cur_step is None or cur_step.get('action') not in ('TURN_LEFT', 'TURN_RIGHT'):
            return 0.0

        boundaries = {'START', 'FLOOR_CHANGE', 'ARRIVE'}
        seg_start = 0
        seg_end = len(steps)
        for i, s in enumerate(steps):
            sn = s.get('step_number', 0)
            act = s.get('action', '')
            if act in boundaries and sn < step_num:
                seg_start = i
            if act in boundaries and sn > step_num and i < seg_end:
                seg_end = i
                break

        turn_count = 0
        for s in steps[seg_start:seg_end + 1]:
            if s.get('action') in ('TURN_LEFT', 'TURN_RIGHT'):
                turn_count += 1
        return 1.0 if turn_count == 1 else 0.0

    @staticmethod
    def _dist_to_adjacent_turn(step_num: int, steps: List[Dict], direction: int) -> float:
        """
        Cumulative metre distance to the previous (direction=-1) or
        next (direction=+1) TURN_LEFT/TURN_RIGHT step.
        Normalised to [0, 1] by MAX_DISTANCE_M.  1.0 if none found.
        """
        ordered = sorted(steps, key=lambda s: s.get('step_number', 0))
        cur_idx = -1
        for i, s in enumerate(ordered):
            if s.get('step_number', 0) == step_num:
                cur_idx = i
                break
        if cur_idx < 0:
            return 1.0

        total_m = 0.0
        if direction < 0:
            for i in range(cur_idx - 1, -1, -1):
                total_m += ordered[i].get('distance_meters', 0.0)
                if ordered[i].get('action') in ('TURN_LEFT', 'TURN_RIGHT'):
                    return min(total_m / MAX_DISTANCE_M, 1.0)
        else:
            for i in range(cur_idx + 1, len(ordered)):
                total_m += ordered[i].get('distance_meters', 0.0)
                if ordered[i].get('action') in ('TURN_LEFT', 'TURN_RIGHT'):
                    return min(total_m / MAX_DISTANCE_M, 1.0)
        return 1.0

    # ─────────────────────────────────────────────────────────────
    #  ANCHOR SELECTION FEATURES  (24-dim per candidate)
    #
    #  Layout per candidate (24 values):
    #    A: Per-candidate  [0..10]   11 numeric
    #    B: Turn context   [11..16]   6 numeric (same for all 3)
    #    C: Comparative    [17..22]   6 numeric (relative to peers)
    #    type_idx           [23]       1 int    (embedding index)
    # ─────────────────────────────────────────────────────────────

    def extract_anchor_features(self, route_record: Dict) -> List[Dict]:
        """
        For each turn that has 3 anchor candidates, produce a dict:
            {
                'features': np.ndarray of shape (3, ANCHOR_RAW_DIM),
                'step_number': int,
                'candidates': [(type, id), ...],
            }
        Returns a list of such dicts (one per eligible turn/step).
        """
        steps = route_record['steps']
        turns = route_record.get('turns', [])
        path_points = route_record.get('path_points', [])
        total_dist = route_record.get('total_distance', 1.0) or 1.0

        turn_by_index = self._index_turns(turns, steps, path_points)

        results = []
        for step in steps:
            sn = step.get('step_number', 0)
            turn = turn_by_index.get(sn)
            if not turn:
                continue

            primary = turn.get('anchor')
            if not primary:
                continue
            alt_list = primary[4] if len(primary) > 4 else []

            candidates = [primary[:4]]
            for alt in alt_list:
                candidates.append(alt[:4])
            while len(candidates) < 3:
                candidates.append(candidates[-1])
            candidates = candidates[:3]

            seg_before = step.get('distance_meters', 0.0)
            seg_after = self._segment_after(sn, steps)
            incoming = turn.get('_incoming_vector')
            outgoing = turn.get('_outgoing_vector')
            cum_dist = step.get('cumulative_distance', 0.0)

            # A: per-candidate raw features (without context/comparative)
            raw_per_cand = []
            for cand in candidates:
                raw_per_cand.append(
                    self._per_candidate_features(
                        cand, turn, seg_before, seg_after,
                        incoming, outgoing,
                    )
                )

            # B: turn context (shared across all 3 candidates)
            ctx = self._turn_context_features(turn, cum_dist, total_dist)

            # C: comparative features (require all 3 candidates' raw data)
            comp = self._comparative_features(raw_per_cand)

            # Assemble final vectors: A + B + C + type_idx
            feat_rows = []
            for i, per_cand in enumerate(raw_per_cand):
                numeric_a = per_cand['numeric']     # 11 values
                type_idx = per_cand['type_idx']      # 1 int
                row = numeric_a + ctx + comp[i] + [float(type_idx)]
                feat_rows.append(row)

            results.append({
                'features': np.array(feat_rows, dtype=np.float32),
                'step_number': sn,
                'candidates': [(c[0], c[1]) for c in candidates],
            })

        return results

    # ── Layer A: Per-Candidate Features (11 numeric + type_idx) ──

    def _per_candidate_features(
        self,
        candidate: Tuple,
        turn: Dict,
        seg_before: float,
        seg_after: float,
        incoming: Optional[Tuple],
        outgoing: Optional[Tuple],
    ) -> Dict:
        """
        Compute 11 numeric features + type_idx for a single candidate.
        Returns dict with 'numeric' (list[float]) and 'type_idx' (int),
        plus raw values needed by comparative features.
        """
        room_type, room_id, area, dist_px = candidate[:4]
        turn_point = turn['point']

        dist_m = dist_px * self.px2m

        # 1) distance_to_turn normalised
        dist_norm = min(dist_m / MAX_ANCHOR_DIST_M, 1.0)

        # 2) distance_score from anchor_selector heuristic (stepped 0.05-1.0)
        d_score = get_distance_score(dist_px, self.px2m)

        # 3) size_score from anchor_selector (0.05-1.0)
        s_score = get_size_score(area)

        # 4) visibility_score
        vis = 0.0
        if turn.get('_visibility_score') is not None:
            vis = turn['_visibility_score']

        # 5) tier (normalised: 1->1.0, 2->0.8, 3->0.5, 4->0.0, 5->0.3)
        raw_tier = get_anchor_tier(room_type)
        tier_map = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.0, 5: 0.3}
        tier_norm = tier_map.get(raw_tier, 0.3)

        # 6) same_side_flag
        same_side = 0.0
        anchor_side = turn.get('anchor_side')
        direction = turn.get('direction', '')
        if anchor_side and direction:
            side_map = {'sol': 'sola', 'sag': 'sağa'}
            if side_map.get(anchor_side) == direction:
                same_side = 1.0

        # 7) relative_position_to_turn
        rel_pos = 0.0
        if turn.get('_relative_anchor_position') is not None:
            rel_pos = turn['_relative_anchor_position']

        # 8) angle_alignment_with_turn
        alignment = 0.0
        if outgoing and turn.get('_anchor_center'):
            alignment = _angle_alignment(
                turn_point, turn['_anchor_center'], outgoing
            )

        # 9) anchor_type index (for embedding)
        type_idx = _room_type_index(room_type)

        # 10-11) segment lengths normalised
        seg_b = min(seg_before / MAX_DISTANCE_M, 1.0)
        seg_a = min(seg_after / MAX_DISTANCE_M, 1.0)

        numeric = [
            dist_norm,     # 0
            d_score,       # 1
            s_score,       # 2
            vis,           # 3
            tier_norm,     # 4
            same_side,     # 5
            rel_pos,       # 6
            alignment,     # 7
            seg_b,         # 8
            seg_a,         # 9
            dist_m / MAX_ANCHOR_DIST_M,  # 10: raw distance ratio (for comparative)
        ]

        return {
            'numeric': numeric,
            'type_idx': type_idx,
            'raw_dist': dist_norm,
            'raw_vis': vis,
        }

    # ── Layer B: Turn Context Features (6 values, same for all candidates) ──

    def _turn_context_features(
        self,
        turn: Dict,
        cumulative_distance: float,
        total_distance: float,
    ) -> List[float]:
        """6-dim context vector describing the turn itself."""
        # 1) turn_angle normalised [0, 1]
        turn_angle = turn.get('angle', 0.0) / 180.0

        # 2-4) turn_type one-hot: [turn, bend, veer]
        tt = turn.get('turn_type', 'turn')
        tt_oh = [0.0, 0.0, 0.0]
        if tt in TURN_TYPES:
            tt_oh[TURN_TYPES.index(tt)] = 1.0
        else:
            tt_oh[0] = 1.0

        # 5) turn_direction: -1=sola, 0=duz, +1=saga
        direction = turn.get('direction', '')
        if direction == 'sola':
            t_dir = -1.0
        elif direction in ('sağa', 'saga'):
            t_dir = 1.0
        else:
            t_dir = 0.0

        # 6) path_ratio: where in the route are we?
        path_ratio = 0.0
        if total_distance > 0:
            path_ratio = min(cumulative_distance / total_distance, 1.0)

        return [turn_angle] + tt_oh + [t_dir, path_ratio]

    # ── Layer C: Comparative Features (6 values per candidate) ──

    @staticmethod
    def _comparative_features(
        raw_per_cand: List[Dict],
    ) -> List[List[float]]:
        """
        Given 3 candidates' raw data, compute 6 comparative features
        for each candidate relative to the others.
        Returns list of 3 lists, each with 6 floats.
        """
        dists = [c['raw_dist'] for c in raw_per_cand]
        vises = [c['raw_vis'] for c in raw_per_cand]

        min_dist = min(dists)
        max_dist = max(dists) if max(dists) > 0 else 1.0
        max_vis = max(vises) if max(vises) > 0 else 1.0

        dist_sorted = sorted(range(3), key=lambda i: dists[i])
        vis_sorted = sorted(range(3), key=lambda i: -vises[i])

        dist_rank = [0.0] * 3
        vis_rank = [0.0] * 3
        for rank, idx in enumerate(dist_sorted):
            dist_rank[idx] = (rank + 1) / 3.0
        for rank, idx in enumerate(vis_sorted):
            vis_rank[idx] = (rank + 1) / 3.0

        result = []
        for i in range(3):
            is_closest = 1.0 if dists[i] <= min_dist + 1e-6 else 0.0
            is_most_vis = 1.0 if vises[i] >= max_vis - 1e-6 else 0.0
            dist_gap = (dists[i] - min_dist) / max_dist if max_dist > 0 else 0.0
            vis_gap = (max_vis - vises[i]) / max_vis if max_vis > 0 else 0.0

            result.append([
                dist_rank[i],    # 0: distance rank (1/3 = closest)
                vis_rank[i],     # 1: visibility rank (1/3 = most visible)
                is_closest,      # 2: binary flag
                is_most_vis,     # 3: binary flag
                dist_gap,        # 4: gap to best distance
                vis_gap,         # 5: gap to best visibility
            ])

        return result

    # ─────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────

    def _index_turns(
        self,
        turns: List[Dict],
        steps: List[Dict],
        path_points: List[Tuple],
    ) -> Dict[int, Dict]:
        """
        Map each step_number to its corresponding turn dict.
        Steps 1 (START) and last (ARRIVE) have no turn.
        Step 2 -> turns[0], step 3 -> turns[1], etc.
        
        Also enriches each turn dict with:
          _incoming_vector, _outgoing_vector, _visibility_score,
          _relative_anchor_position, _anchor_center
        """
        mapping: Dict[int, Dict] = {}
        for i, turn in enumerate(turns):
            step_num = i + 2  # turns[0] -> step 2
            if step_num > len(steps):
                break

            pi = turn.get('path_index', 0)
            if pi > 0 and pi < len(path_points) - 1:
                prev_pt = path_points[pi - 1]
                curr_pt = path_points[pi]
                next_pt = path_points[pi + 1]
                incoming = (curr_pt[0] - prev_pt[0], curr_pt[1] - prev_pt[1])
                outgoing = (next_pt[0] - curr_pt[0], next_pt[1] - curr_pt[1])
            else:
                incoming = None
                outgoing = None

            turn['_incoming_vector'] = incoming
            turn['_outgoing_vector'] = outgoing

            anchor = turn.get('anchor')
            anchor_center = None
            vis_score = 0.0
            rel_pos = 0.0

            if anchor and incoming:
                anchor_center = turn.get('_anchor_center')

                if anchor_center:
                    _, view_score = is_in_view_cone(
                        turn_point=turn['point'],
                        anchor_center=anchor_center,
                        incoming_vector=incoming,
                        outgoing_vector=outgoing,
                    )
                    vis_score = view_score
                    rel_pos = _compute_relative_anchor_position(
                        turn['point'], anchor_center, incoming
                    )
                else:
                    vis_score = 0.5

            turn['_visibility_score'] = vis_score
            turn['_relative_anchor_position'] = rel_pos
            if anchor_center:
                turn['_anchor_center'] = anchor_center

            mapping[step_num] = turn

        return mapping

    @staticmethod
    def _segment_after(step_num: int, steps: List[Dict]) -> float:
        """Distance of the next step segment (metres)."""
        for s in steps:
            if s.get('step_number', 0) == step_num + 1:
                return s.get('distance_meters', 0.0)
        return 0.0

    # ─────────────────────────────────────────────
    #  Description Strategy Features (16-dim)
    # ─────────────────────────────────────────────

    def extract_strategy_features(self, route_record: Dict) -> np.ndarray:
        """
        Return (N_steps, 16) float32 array of strategy features.

        These features help predict whether a human would describe a step
        using an anchor reference or use a non-anchor description style
        (proximity, orientation, geometry).

        Layout per step:
          [0..3]   route context  (4 dims)
          [4..9]   step context   (6 dims)
          [10..15] spatial        (6 dims)
        """
        steps = route_record['steps']
        turns = route_record.get('turns', [])
        path_points = route_record.get('path_points', [])
        total_dist = route_record.get('total_distance', 1.0) or 1.0

        start_room = route_record.get('start_room', {})
        end_room = route_record.get('end_room', {})
        start_center = start_room.get('center')
        end_center = end_room.get('center')

        turn_by_index = self._index_turns(turns, steps, path_points)

        route_ctx = self._route_context_features(
            total_dist, turns, steps, start_center, end_center,
        )

        features = []
        for step in steps:
            step_ctx = self._step_context_features(step, steps, turn_by_index, total_dist)
            spatial = self._spatial_features(
                step, turn_by_index, path_points,
                start_center, end_center, total_dist,
            )
            features.append(route_ctx + step_ctx + spatial)

        return np.array(features, dtype=np.float32)

    # -- helpers for strategy features --

    def _route_context_features(
        self,
        total_dist: float,
        turns: List[Dict],
        steps: List[Dict],
        start_center: Optional[Tuple],
        end_center: Optional[Tuple],
    ) -> List[float]:
        """4-dim route-level context shared by all steps."""
        total_norm = min(total_dist / MAX_DISTANCE_M, 1.0)
        turn_norm = min(len(turns) / 10.0, 1.0)
        step_norm = min(len(steps) / 10.0, 1.0)

        direct_dist = 0.0
        if start_center and end_center:
            dx = end_center[0] - start_center[0]
            dy = end_center[1] - start_center[1]
            direct_dist = math.sqrt(dx * dx + dy * dy) * self.px2m
        direct_norm = min(direct_dist / MAX_DISTANCE_M, 1.0)

        return [total_norm, turn_norm, step_norm, direct_norm]

    @staticmethod
    def _step_context_features(
        step: Dict,
        steps: List[Dict],
        turn_by_index: Dict[int, Dict],
        total_dist: float,
    ) -> List[float]:
        """6-dim step-level context."""
        action = step.get('action', 'START')
        sn = step.get('step_number', 0)

        action_norm = ACTION_TO_IDX.get(action, 0) / max(len(ACTION_TYPES) - 1, 1)
        seg_len = step.get('distance_meters', 0.0)
        seg_ratio = seg_len / total_dist if total_dist > 0 else 0.0
        cum = step.get('cumulative_distance', 0.0)
        cum_ratio = cum / total_dist if total_dist > 0 else 0.0
        is_first = 1.0 if sn == 1 else 0.0
        is_last = 1.0 if action == 'ARRIVE' else 0.0
        has_turn = 1.0 if sn in turn_by_index else 0.0

        return [action_norm, min(seg_ratio, 1.0), min(cum_ratio, 1.0),
                is_first, is_last, has_turn]

    def _spatial_features(
        self,
        step: Dict,
        turn_by_index: Dict[int, Dict],
        path_points: List[Tuple],
        start_center: Optional[Tuple],
        end_center: Optional[Tuple],
        total_dist: float,
    ) -> List[float]:
        """
        6-dim spatial features capturing why humans might skip anchors.

        [room_behind, room_side, nearest_anchor_dist, anchor_vis,
         junction_degree, rooms_adjacent]
        """
        action = step.get('action', '')
        sn = step.get('step_number', 0)
        turn = turn_by_index.get(sn)

        room_behind = 0.0
        room_side = 0.0
        nearest_anchor_dist = 1.0
        anchor_vis = 0.0
        junction_degree = 0.0
        rooms_adjacent = 0.0

        # -- room_behind & room_side (for START steps) --
        direction = step.get('direction', '')
        if action == 'START':
            if direction == 'arka':
                room_behind = 1.0
                room_side = 0.0
            elif direction == 'sol':
                room_side = -1.0
            elif direction == 'sag':
                room_side = 1.0

        # -- nearest anchor distance & visibility --
        if turn and turn.get('anchor'):
            raw_dist = turn['anchor'][3] * self.px2m
            nearest_anchor_dist = min(raw_dist / MAX_ANCHOR_DIST_M, 1.0)
            anchor_vis = turn.get('_visibility_score', 0.0)
        elif action == 'START' and start_center and len(path_points) >= 2:
            anchor_vis = self._estimate_room_visibility(
                start_center, path_points[0], path_points[1],
            )
        elif action == 'ARRIVE' and end_center and len(path_points) >= 2:
            anchor_vis = self._estimate_room_visibility(
                end_center, path_points[-1], path_points[-2],
            )

        # -- junction degree (approximate from path_points) --
        if turn:
            pi = turn.get('path_index', 0)
            if 0 < pi < len(path_points):
                junction_degree = self._approx_junction_degree(
                    path_points[pi], path_points,
                )
        elif action == 'ARRIVE' and len(path_points) > 0:
            junction_degree = self._approx_junction_degree(
                path_points[-1], path_points,
            )

        # -- rooms adjacent --
        if start_center and end_center:
            dx = end_center[0] - start_center[0]
            dy = end_center[1] - start_center[1]
            direct_m = math.sqrt(dx * dx + dy * dy) * self.px2m
            if direct_m < ADJACENT_THRESHOLD_M:
                rooms_adjacent = 1.0

        return [room_behind, room_side, nearest_anchor_dist,
                anchor_vis, junction_degree, rooms_adjacent]

    @staticmethod
    def _estimate_room_visibility(
        room_center: Tuple,
        step_point: Tuple,
        next_point: Tuple,
    ) -> float:
        """Estimate visibility of a room from a path point using dot product."""
        walk_dx = next_point[0] - step_point[0]
        walk_dy = next_point[1] - step_point[1]
        walk_mag = math.sqrt(walk_dx * walk_dx + walk_dy * walk_dy)
        if walk_mag < 1e-6:
            return 0.5

        room_dx = room_center[0] - step_point[0]
        room_dy = room_center[1] - step_point[1]
        room_mag = math.sqrt(room_dx * room_dx + room_dy * room_dy)
        if room_mag < 1e-6:
            return 1.0

        dot = (walk_dx * room_dx + walk_dy * room_dy) / (walk_mag * room_mag)
        return max((dot + 1.0) / 2.0, 0.0)

    @staticmethod
    def _approx_junction_degree(
        point: Tuple,
        path_points: List[Tuple],
        threshold: float = 5.0,
    ) -> float:
        """
        Approximate junction degree: count how many distinct path segments
        pass near a given point.  Normalized to [0, 1] by dividing by 6.
        """
        count = 0
        for i in range(len(path_points) - 1):
            p = path_points[i]
            dx = p[0] - point[0]
            dy = p[1] - point[1]
            if math.sqrt(dx * dx + dy * dy) < threshold:
                count += 1
        return min(count / 6.0, 1.0)
