"""
mT5-small Model Wrapper for Route Description Generation

Handles model loading, checkpoint management, and text generation
with beam search.  Runs on CPU.
"""
import os
from typing import List, Optional

MODEL_NAME = 'google/mt5-small'
DEFAULT_CHECKPOINT_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'checkpoints', 'text_gen_best',
)


class RouteDescriptionGenerator:
    """
    Wraps HuggingFace mT5-small for route description generation.

    Usage:
        gen = RouteDescriptionGenerator()
        gen.load()  # loads pretrained or fine-tuned checkpoint
        text = gen.generate("describe: action=TURN_RIGHT | dist=7.1 | ...")
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        checkpoint_dir: str = None,
        device: str = 'cpu',
    ):
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
        self.device = device
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self, from_checkpoint: bool = True) -> bool:
        """
        Load the model and tokenizer.

        Args:
            from_checkpoint: If True, try loading from fine-tuned checkpoint
                             first, fall back to pretrained.

        Returns:
            True if a fine-tuned checkpoint was loaded, False if pretrained.
        """
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        loaded_finetuned = False

        if from_checkpoint and os.path.isdir(self.checkpoint_dir):
            config_path = os.path.join(self.checkpoint_dir, 'config.json')
            if os.path.exists(config_path):
                print(f"[TextGen] Loading fine-tuned model from {self.checkpoint_dir}")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.checkpoint_dir
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.checkpoint_dir
                )
                loaded_finetuned = True

        if not loaded_finetuned:
            print(f"[TextGen] Loading pretrained {self.model_name}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[TextGen] Model loaded ({param_count / 1e6:.1f}M params, device={self.device})")
        return loaded_finetuned

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def generate(
        self,
        input_text: str,
        max_length: int = 64,
        num_beams: int = 4,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate a description from a formatted input string.

        Returns the top-scoring sequence as a string.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        inputs = self.tokenizer(
            input_text,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                early_stopping=True,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.strip()

    def generate_batch(
        self,
        input_texts: List[str],
        max_length: int = 64,
        num_beams: int = 4,
    ) -> List[str]:
        """Generate descriptions for a batch of inputs."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import torch

        inputs = self.tokenizer(
            input_texts,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        return [
            self.tokenizer.decode(o, skip_special_tokens=True).strip()
            for o in outputs
        ]

    def save_checkpoint(self, path: str = None):
        """Save model and tokenizer to a directory."""
        path = path or self.checkpoint_dir
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[TextGen] Checkpoint saved to {path}")
