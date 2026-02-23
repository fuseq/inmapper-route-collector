"""
Text Generation Dataset

Builds input/target text pairs from route records and tokenizes them
for mT5 fine-tuning using the HuggingFace tokenizer.
"""
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ml.text_gen.formatter import format_training_pair

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 64


class TextGenDataset(Dataset):
    """
    PyTorch Dataset for mT5 fine-tuning.

    Each sample is a (input_ids, attention_mask, labels) tuple where:
      - input_ids: tokenized structured step description
      - labels: tokenized human description
    """

    def __init__(
        self,
        route_records: List[Dict],
        tokenizer,
        max_input_length: int = MAX_INPUT_LENGTH,
        max_target_length: int = MAX_TARGET_LENGTH,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.samples: List[Tuple[str, str]] = []

        self._build_pairs(route_records)

    def _build_pairs(self, route_records: List[Dict]):
        """Extract kept step pairs from all route records."""
        from ml.data.data_loader import parse_human_steps, extract_step_labels

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
                    self.samples.append(pair)

        print(f"[TextGenDataset] Built {len(self.samples)} training pairs")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_text, target_text = self.samples[idx]

        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        labels = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        label_ids = labels['input_ids'].squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': model_inputs['input_ids'].squeeze(0),
            'attention_mask': model_inputs['attention_mask'].squeeze(0),
            'labels': label_ids,
        }

    def raw_pairs(self) -> List[Tuple[str, str]]:
        """Return the raw (input, target) text pairs."""
        return list(self.samples)


def build_text_gen_pairs(route_records: List[Dict]) -> List[Tuple[str, str]]:
    """
    Build (input_text, target_text) pairs without tokenization.
    Useful for inspection and evaluation.
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
