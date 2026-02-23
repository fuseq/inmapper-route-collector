"""
Text Generation Training Script

Fine-tunes mT5-small on (structured input -> human description) pairs
collected from route annotations.

Usage:
    python -m ml.text_gen.train --source local --epochs 20
    python -m ml.text_gen.train --source local --epochs 20 --lr 3e-4
"""
import argparse
import os
import sys

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, PROJECT_ROOT)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'text_gen_best')


def train_text_gen(
    train_records: list,
    val_records: list,
    model_name: str = 'google/mt5-small',
    lr: float = 3e-4,
    batch_size: int = 8,
    epochs: int = 20,
    warmup_steps: int = 100,
    max_input_length: int = 128,
    max_target_length: int = 64,
) -> dict:
    """Fine-tune mT5-small and return training results."""
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )

    from ml.text_gen.dataset import TextGenDataset

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    output_dir = os.path.join(CHECKPOINT_DIR, 'runs')
    os.makedirs(output_dir, exist_ok=True)

    print(f"[TextGen Train] Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_ds = TextGenDataset(
        train_records, tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )
    val_ds = TextGenDataset(
        val_records, tokenizer,
        max_input_length=max_input_length,
        max_target_length=max_target_length,
    )

    print(f"[TextGen Train] Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    if len(train_ds) == 0:
        print("[TextGen Train] No training data. Skipping.")
        return {}

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        logging_steps=10,
        fp16=False,
        use_cpu=True,
        report_to='none',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_ds) > 0 else None,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("[TextGen Train] Starting fine-tuning...")
    train_result = trainer.train()

    model.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    print(f"[TextGen Train] Best model saved to {CHECKPOINT_DIR}")

    metrics = train_result.metrics
    print(f"[TextGen Train] Training loss: {metrics.get('train_loss', 'N/A'):.4f}")

    if len(val_ds) > 0:
        eval_metrics = trainer.evaluate()
        print(f"[TextGen Train] Eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
        metrics.update(eval_metrics)

    _print_sample_predictions(model, tokenizer, val_ds if len(val_ds) > 0 else train_ds)

    return metrics


def _print_sample_predictions(model, tokenizer, dataset, n_samples: int = 3):
    """Print a few sample predictions for quick inspection."""
    import torch

    model.eval()
    n = min(n_samples, len(dataset))
    if n == 0:
        return

    print("\n[TextGen Train] Sample predictions:")
    for i in range(n):
        sample = dataset[i]
        input_ids = sample['input_ids'].unsqueeze(0)
        labels = sample['labels'].clone()
        labels[labels == -100] = tokenizer.pad_token_id

        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        target_text = tokenizer.decode(labels, skip_special_tokens=True)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_length=64,
                num_beams=4,
                early_stopping=True,
            )
        pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"  Input:  {input_text[:80]}...")
        print(f"  Target: {target_text}")
        print(f"  Pred:   {pred_text}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Fine-tune mT5 for route descriptions')
    parser.add_argument(
        '--source', choices=['local', 'sheets'],
        default='local', help='Data source',
    )
    parser.add_argument('--sheets-url', type=str, default='')
    parser.add_argument('--venue', type=str, default='zorlu')
    parser.add_argument('--model-name', type=str, default='google/mt5-small')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    from ml.data.data_loader import build_training_dataset
    from ml.data.dataset import train_val_split

    records = build_training_dataset(
        source=args.source,
        sheets_url=args.sheets_url,
        venue=args.venue,
        base_dir=PROJECT_ROOT,
    )

    if not records:
        print("[TextGen Train] No data available. Exiting.")
        return

    _store_raw_human_text(records, args.source, args.sheets_url)

    train_recs, val_recs = train_val_split(
        records, val_ratio=args.val_ratio, seed=args.seed,
    )
    print(f"[TextGen Train] Split: {len(train_recs)} train / {len(val_recs)} val routes")

    results = train_text_gen(
        train_recs, val_recs,
        model_name=args.model_name,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
    )

    print("\n[TextGen Train] Done.")
    return results


def _store_raw_human_text(records: list, source: str, sheets_url: str):
    """
    Attach the raw human_steps text to each record so the dataset
    builder can parse it.  build_training_dataset() doesn't store
    the raw text by default, so we reload and match by route_id.
    """
    from ml.data.data_loader import (
        load_from_local_submissions,
        load_from_google_sheets,
    )

    if source == 'sheets' and sheets_url:
        raw_rows = load_from_google_sheets(sheets_url)
    else:
        raw_rows = load_from_local_submissions()

    text_by_id = {}
    for row in raw_rows:
        rid = row.get('id', '')
        if rid:
            text_by_id[rid] = row.get('human_steps', '')

    matched = 0
    for rec in records:
        rid = rec.get('route_id', '')
        if rid in text_by_id:
            rec['_raw_human_text'] = text_by_id[rid]
            matched += 1

    print(f"[TextGen Train] Matched raw human text for {matched}/{len(records)} records")


if __name__ == '__main__':
    main()
