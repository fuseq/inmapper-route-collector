"""
Navigation Decision Pipeline

End-to-end inference: takes raw route generation output, applies the
trained classification models, and generates natural descriptions.

Flow:
    Step -> StepFilter -> DescriptionStrategy -> AnchorSelection -> TextGeneration

Usage:
    pipeline = NavigationDecisionPipeline(
        step_filter_path='ml/checkpoints/step_filter_best.pt',
        anchor_selector_path='ml/checkpoints/anchor_selector_best.pt',
        strategy_path='ml/checkpoints/description_strategy_best.pt',
        text_gen_dir='ml/checkpoints/text_gen_best',
    )
    result = pipeline.predict(route_data)
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.features.feature_extractor import (
    FeatureExtractor, ANCHOR_TYPE_IDX_COL, ANCHOR_NUMERIC_DIM,
)
from ml.models.step_filter import StepFilterModel
from ml.models.anchor_selector import AnchorSelectionModel
from ml.models.description_strategy import DescriptionStrategyModel


class NavigationDecisionPipeline:
    """
    Loads trained StepFilter, DescriptionStrategy, AnchorSelection, and
    optional TextGeneration models, then applies them sequentially.
    """

    def __init__(
        self,
        step_filter_path: Optional[str] = None,
        anchor_selector_path: Optional[str] = None,
        strategy_path: Optional[str] = None,
        text_gen_dir: Optional[str] = None,
        step_filter_threshold: float = 0.50,
        strategy_threshold: float = 0.5,
        use_text_gen: bool = False,
    ):
        self.fe = FeatureExtractor()
        self.threshold = step_filter_threshold
        self.strategy_threshold = strategy_threshold

        self.step_filter = None
        self.anchor_selector = None
        self.strategy_model = None
        self.text_gen = None

        if step_filter_path and os.path.exists(step_filter_path):
            self.step_filter = StepFilterModel()
            self.step_filter.load_state_dict(
                torch.load(step_filter_path, map_location='cpu', weights_only=True)
            )
            self.step_filter.eval()
            print(f"[Pipeline] Loaded StepFilter from {step_filter_path}")

        if strategy_path and os.path.exists(strategy_path):
            self.strategy_model = DescriptionStrategyModel()
            self.strategy_model.load_state_dict(
                torch.load(strategy_path, map_location='cpu', weights_only=True)
            )
            self.strategy_model.eval()
            print(f"[Pipeline] Loaded DescriptionStrategy from {strategy_path}")

        if anchor_selector_path and os.path.exists(anchor_selector_path):
            self.anchor_selector = AnchorSelectionModel()
            self.anchor_selector.load_state_dict(
                torch.load(anchor_selector_path, map_location='cpu', weights_only=True)
            )
            self.anchor_selector.eval()
            print(f"[Pipeline] Loaded AnchorSelector from {anchor_selector_path}")

        if use_text_gen and text_gen_dir and os.path.isdir(text_gen_dir):
            config_path = os.path.join(text_gen_dir, 'config.json')
            if os.path.exists(config_path):
                try:
                    from ml.text_gen.model import RouteDescriptionGenerator
                    self.text_gen = RouteDescriptionGenerator(checkpoint_dir=text_gen_dir)
                    self.text_gen.load(from_checkpoint=True)
                except Exception as e:
                    print(f"[Pipeline] Failed to load TextGen: {e}")
                    self.text_gen = None

    def predict(self, route_data: Dict) -> Dict:
        """
        Apply all four stages to a single route.

        Flow:
          1. StepFilter: predict keep/delete for every step
          2. DescriptionStrategy: for each kept step, predict anchor-based vs non-anchor
          3. AnchorSelection: for anchor-based steps that have a turn, select best anchor
          4. TextGeneration: generate natural descriptions for kept steps

        Args:
            route_data: dict with keys: steps, turns, path_points, total_distance
                        (as returned by regenerate_route or route generation)

        Returns:
            {
                'filtered_steps': list of steps predicted as KEEP,
                'all_step_decisions': [{step_number, action, keep_prob, keep}, ...],
                'strategy_decisions': [{step_number, anchor_prob, anchor_based}, ...],
                'anchor_decisions': [{step_number, selected_idx, proba, candidates}, ...],
                'generated_descriptions': [{step_number, description, source}, ...],
            }
        """
        result = {
            'filtered_steps': [],
            'all_step_decisions': [],
            'strategy_decisions': [],
            'anchor_decisions': [],
            'generated_descriptions': [],
        }

        steps = route_data.get('steps', [])
        if not steps:
            return result

        # ── Step Filtering ──
        step_features = self.fe.extract_step_features(route_data)

        if self.step_filter is not None:
            X = torch.from_numpy(step_features)
            with torch.no_grad():
                logits = self.step_filter(X)
                proba = torch.sigmoid(logits).numpy()
                keep = (proba >= self.threshold).astype(int)
        else:
            proba = np.ones(len(steps), dtype=np.float32)
            keep = np.ones(len(steps), dtype=int)

        for i, step in enumerate(steps):
            action = step.get('action', '')
            # FLOOR_CHANGE and START_PORTAL are structural -- always keep them
            if action in ('FLOOR_CHANGE', 'START_PORTAL'):
                keep[i] = 1
            # Portal arrivals are redundant (FLOOR_CHANGE already describes the transition)
            elif action == 'ARRIVE_PORTAL':
                keep[i] = 0
            elif (action == 'ARRIVE'
                  and i + 1 < len(steps)
                  and steps[i + 1].get('action') == 'FLOOR_CHANGE'):
                keep[i] = 0
            # Fallback: START right after FLOOR_CHANGE is a portal exit
            elif (action == 'START'
                  and i - 1 >= 0
                  and steps[i - 1].get('action') == 'FLOOR_CHANGE'):
                keep[i] = 1
                step['action'] = 'START_PORTAL'
                step['portal_type'] = steps[i - 1].get('portal_type', '')

        # Segment-based post-processing: ensure mid-route steps per segment
        self._ensure_segment_navigation(steps, keep, proba)

        kept_indices = set()
        for i, step in enumerate(steps):
            decision = {
                'step_number': step.get('step_number', i + 1),
                'action': step.get('action', ''),
                'keep_prob': float(proba[i]),
                'keep': bool(keep[i]),
            }
            result['all_step_decisions'].append(decision)

            if keep[i]:
                result['filtered_steps'].append(step)
                kept_indices.add(i)

        # ── Description Strategy ──
        strategy_features = self.fe.extract_strategy_features(route_data)
        anchor_based_steps = set()

        for i, step in enumerate(steps):
            if i not in kept_indices:
                continue

            sn = step.get('step_number', i + 1)

            if self.strategy_model is not None:
                X_s = torch.from_numpy(strategy_features[i]).unsqueeze(0)
                with torch.no_grad():
                    logit = self.strategy_model(X_s)
                    anchor_prob = float(torch.sigmoid(logit).item())
                    is_anchor = anchor_prob >= self.strategy_threshold
            else:
                anchor_prob = 1.0
                is_anchor = True

            result['strategy_decisions'].append({
                'step_number': sn,
                'anchor_prob': anchor_prob,
                'anchor_based': is_anchor,
            })

            if is_anchor:
                anchor_based_steps.add(sn)

        # ── Anchor Selection (only for anchor-based steps with turns) ──
        anchor_records = self.fe.extract_anchor_features(route_data)

        for arec in anchor_records:
            sn = arec['step_number']
            if sn not in anchor_based_steps:
                continue

            feat = arec['features']  # (3, ANCHOR_RAW_DIM)
            type_col = feat[:, ANCHOR_TYPE_IDX_COL].astype(np.int64)
            numeric_feat = np.delete(feat, ANCHOR_TYPE_IDX_COL, axis=1).astype(np.float32)

            if self.anchor_selector is not None:
                num_t = torch.from_numpy(numeric_feat).unsqueeze(0)
                type_t = torch.from_numpy(type_col).unsqueeze(0)
                with torch.no_grad():
                    logits = self.anchor_selector(num_t, type_t)
                    proba_a = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                    selected = int(logits.argmax(dim=-1).item())
            else:
                proba_a = np.array([1.0, 0.0, 0.0])
                selected = 0

            result['anchor_decisions'].append({
                'step_number': sn,
                'selected_idx': selected,
                'proba': proba_a.tolist(),
                'candidates': arec['candidates'],
                'selected_anchor': arec['candidates'][selected],
            })

        # ── Text Generation ──
        result['generated_descriptions'] = self._generate_descriptions(
            route_data, result, strategy_features,
        )

        return result

    @staticmethod
    def _ensure_segment_navigation(
        steps: List[Dict],
        keep: np.ndarray,
        proba: np.ndarray,
    ) -> None:
        """
        Post-processing safety net for each navigation segment (bounded by
        START / START_PORTAL / FLOOR_CHANGE / ARRIVE / ARRIVE_PORTAL):

        1. If no TURN step was kept, force-keep the best-probability TURN.
        2. If still no mid-route step at all (no TURN, VEER, PASS_BY kept),
           force-keep the best-probability mid-route step of any type.

        This guarantees every segment has at least one navigation instruction.
        """
        boundaries = {'START', 'START_PORTAL', 'FLOOR_CHANGE', 'ARRIVE', 'ARRIVE_PORTAL'}
        turn_actions = {'TURN_LEFT', 'TURN_RIGHT'}
        mid_actions = {'TURN_LEFT', 'TURN_RIGHT', 'VEER', 'PASS_BY'}

        segments: List[List[int]] = []
        current_seg: List[int] = []

        for i, step in enumerate(steps):
            action = step.get('action', '')
            current_seg.append(i)
            if action in boundaries and i > 0:
                segments.append(current_seg)
                current_seg = [i]
        if current_seg:
            segments.append(current_seg)

        for seg_indices in segments:
            # Pass 1: ensure at least one TURN is kept
            has_kept_turn = False
            best_turn_idx = -1
            best_turn_prob = -1.0

            for idx in seg_indices:
                action = steps[idx].get('action', '')
                if action not in turn_actions:
                    continue
                if keep[idx]:
                    has_kept_turn = True
                    break
                if proba[idx] > best_turn_prob:
                    best_turn_prob = proba[idx]
                    best_turn_idx = idx

            if not has_kept_turn and best_turn_idx >= 0:
                keep[best_turn_idx] = 1

            # Pass 2: if still no mid-route step kept, force-keep best one
            has_any_mid = any(
                keep[idx] and steps[idx].get('action', '') in mid_actions
                for idx in seg_indices
            )
            if has_any_mid:
                continue

            best_mid_idx = -1
            best_mid_prob = -1.0
            for idx in seg_indices:
                action = steps[idx].get('action', '')
                if action not in mid_actions:
                    continue
                if proba[idx] > best_mid_prob:
                    best_mid_prob = proba[idx]
                    best_mid_idx = idx

            if best_mid_idx >= 0:
                keep[best_mid_idx] = 1

    def _generate_descriptions(
        self,
        route_data: Dict,
        result: Dict,
        strategy_features: np.ndarray,
    ) -> List[Dict]:
        """Generate natural descriptions for all kept steps."""
        from ml.text_gen.formatter import (
            format_step_input, get_ablative_suffix, PLACEHOLDER,
            _anchor_type_from_id, _is_portal_anchor,
        )
        from ml.text_gen.templates import generate_description as template_generate

        steps = route_data.get('steps', [])
        route_context = {
            'total_distance': route_data.get('total_distance', 0),
            'start_room': route_data.get('start_room', {}),
            'end_room': route_data.get('end_room', {}),
        }

        strategy_by_sn = {
            d['step_number']: d for d in result.get('strategy_decisions', [])
        }
        anchor_by_sn = {
            d['step_number']: d for d in result.get('anchor_decisions', [])
        }
        kept_sns = {s.get('step_number', 0) for s in result.get('filtered_steps', [])}

        descriptions = []
        prev_action = ''
        for i, step in enumerate(steps):
            sn = step.get('step_number', i + 1)
            if sn not in kept_sns:
                prev_action = step.get('action', '')
                continue

            strat = strategy_by_sn.get(sn)
            anchor = anchor_by_sn.get(sn)
            sf = strategy_features[i].tolist() if i < len(strategy_features) else None
            action = step.get('action', '')

            # Prevent using the destination as a mid-route anchor
            if action not in ('START', 'ARRIVE', 'FLOOR_CHANGE', 'START_PORTAL'):
                anchor, strat = self._check_destination_anchor(
                    anchor, strat, route_context,
                )

            use_model = (
                self.text_gen is not None
                and action not in ('FLOOR_CHANGE', 'START_PORTAL')
            )

            anchor_name = self._get_real_anchor_name(step, anchor)

            # Also catch destination used via step landmark fallback
            if action not in ('START', 'ARRIVE', 'FLOOR_CHANGE', 'START_PORTAL'):
                end_id = str(route_context.get('end_room', {}).get('id', ''))
                if end_id and anchor_name and end_id in anchor_name:
                    anchor_name = ''
                    anchor = None
                    strat = {'anchor_prob': 0.0, 'anchor_based': False}

            if use_model:
                try:
                    input_text = format_step_input(
                        step, strat, anchor, route_context, sf,
                        prev_action=prev_action,
                    )
                    desc = self.text_gen.generate(input_text)
                    desc = self._postprocess_placeholder(desc, anchor_name)
                    source = 'model'
                except Exception:
                    desc = template_generate(step, strat, anchor, route_context, sf)
                    source = 'template'
            else:
                desc = template_generate(step, strat, anchor, route_context, sf)
                source = 'template'

            descriptions.append({
                'step_number': sn,
                'description': desc,
                'source': source,
            })
            prev_action = action

        return descriptions

    @staticmethod
    def _check_destination_anchor(
        anchor: Optional[Dict],
        strat: Optional[Dict],
        route_context: Dict,
    ) -> tuple:
        """If the selected anchor is the destination, switch to non-anchor."""
        if not anchor or not anchor.get('selected_anchor'):
            return anchor, strat

        sel = anchor['selected_anchor']
        if isinstance(sel, (list, tuple)) and len(sel) >= 2:
            anchor_id = str(sel[1])
        else:
            return anchor, strat

        end_room = route_context.get('end_room', {})
        end_id = str(end_room.get('id', ''))

        if anchor_id and end_id and anchor_id == end_id:
            strat = {'anchor_prob': 0.0, 'anchor_based': False}
            anchor = None

        return anchor, strat

    @staticmethod
    def _get_real_anchor_name(step: Dict, anchor_decision: Optional[Dict]) -> str:
        if anchor_decision and anchor_decision.get('selected_anchor'):
            a = anchor_decision['selected_anchor']
            if isinstance(a, (list, tuple)) and len(a) >= 2:
                return f"{a[0]} - {a[1]}"
        return step.get('landmark', '')

    @staticmethod
    def _postprocess_placeholder(desc: str, anchor_name: str) -> str:
        """Replace [ANCHOR] with the real anchor name and fix suffix."""
        from ml.text_gen.formatter import PLACEHOLDER, get_ablative_suffix

        if PLACEHOLDER not in desc or not anchor_name:
            return desc

        suffix = get_ablative_suffix(anchor_name)

        import re
        desc = re.sub(
            re.escape(PLACEHOLDER) + r"'[a-z]+",
            lambda m: anchor_name + "'" + suffix,
            desc,
            count=0,
        )
        desc = desc.replace(PLACEHOLDER, anchor_name)
        return desc

    def predict_batch(self, route_data_list: List[Dict]) -> List[Dict]:
        """Apply prediction to multiple routes."""
        return [self.predict(rd) for rd in route_data_list]


def create_pipeline(
    checkpoint_dir: str = None,
    threshold: float = 0.50,
    strategy_threshold: float = 0.5,
    use_text_gen: bool = False,
) -> NavigationDecisionPipeline:
    """Convenience factory using default checkpoint paths.

    Args:
        use_text_gen: Enable T5 text generation model. Default False
                      (uses templates). Set True when enough training
                      data is available (300+ sentences).
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')

    sf_path = os.path.join(checkpoint_dir, 'step_filter_best.pt')
    as_path = os.path.join(checkpoint_dir, 'anchor_selector_best.pt')
    ds_path = os.path.join(checkpoint_dir, 'description_strategy_best.pt')
    tg_dir = os.path.join(checkpoint_dir, 'text_gen_best')

    return NavigationDecisionPipeline(
        step_filter_path=sf_path if os.path.exists(sf_path) else None,
        anchor_selector_path=as_path if os.path.exists(as_path) else None,
        strategy_path=ds_path if os.path.exists(ds_path) else None,
        text_gen_dir=tg_dir if os.path.isdir(tg_dir) else None,
        step_filter_threshold=threshold,
        use_text_gen=use_text_gen,
        strategy_threshold=strategy_threshold,
    )
