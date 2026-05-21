"""
Fine-tuning OpenAI Whisper (base) for ASR
Hardware target: single NVIDIA L40S (48 GB VRAM)
Data: ~35 hours of labeled audio

Requirements:
    pip install transformers>=4.36 datasets accelerate soundfile librosa \
                evaluate jiwer tensorboard torch torchaudio

Directory structure expected:
    data/
      train/
        audio1.wav  audio2.wav  ...
        metadata.csv   (columns: file_name, transcription)
      validation/
        audio1.wav  ...
        metadata.csv

metadata.csv example:
    file_name,transcription
    audio1.wav,hello world this is a test
    audio2.wav,the quick brown fox
"""

import os
import json
import logging
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
import evaluate
from datasets import load_dataset, Audio, DatasetDict, Dataset, concatenate_datasets
from jiwer import wer, cer
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────

MODEL_ID        = "openai/whisper-base"
LANGUAGE        = "spanish"
TASK            = "transcribe"       # or "translate"
OUTPUT_DIR      = "./whisper-base-finetuned"
DATA_DIR_SCRIPTED     = "./data/scripted"
DATA_DIR_SPONTANEOUS  = "./data/spontaneous"
SAMPLING_RATE   = 16_000
MAX_AUDIO_LEN_S = 30


# Training hyperparameters (tuned for 35 h of data on an L40S)
TRAIN_BATCH_SIZE  = 32
EVAL_BATCH_SIZE   = 16
GRADIENT_ACC      = 1    # effective batch = 32; increase if you lower batch size
LEARNING_RATE     = 1e-5
WARMUP_STEPS      = 1_000
MAX_STEPS         = 15_000 # ~4-5 epochs
SAVE_STEPS        = 500
EVAL_STEPS        = 500
LOGGING_STEPS     = 25
FP16              = True    # L40S supports FP16; set False to use BF16 instead
BF16              = False   # flip to True + FP16=False if you prefer BF16
METRIC_FOR_BEST   = "wer"   # lower is better → set greater_is_better=False


# ─────────────────────────────────────────────
# 2. Load processor (feature extractor + tokenizer)
# ─────────────────────────────────────────────

def build_processor():
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    tokenizer = WhisperTokenizer.from_pretrained(
        MODEL_ID, language=LANGUAGE, task=TASK
    )
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID, language=LANGUAGE, task=TASK
    )
    return feature_extractor, tokenizer, processor


# ─────────────────────────────────────────────
# 3. Load dataset from local CSV + audio files
# ─────────────────────────────────────────────

def load_split(data_dir: str, tsv_filename: str) -> Dataset:
    """Load a single TSV split from a given data directory."""
    tsv_path = os.path.join(data_dir, tsv_filename)
    df = pd.read_csv(tsv_path, sep="\t", usecols=["path", "sentence"])
    df["path"] = df["path"].apply(
        lambda p: os.path.join(data_dir, "wav", os.path.splitext(p)[0] + ".wav")
    )
    df = df.rename(columns={"sentence": "transcription"})
    df = df.dropna(subset=["transcription", "path"])
    df = df[df["path"].apply(os.path.exists)]
    return Dataset.from_pandas(df, preserve_index=False)


def load_local_dataset() -> DatasetDict:
    """
    Loads scripted and spontaneous datasets separately, then merges them.
    Spontaneous is upsampled x UPSAMPLE_FACTOR to compensate for the imbalance.

    Expected structure:
        data/scripted/
          train.tsv, dev.tsv, test.tsv, wav/
        data/spontaneous/
          train.tsv, dev.tsv, test.tsv, wav/
    """
    splits = {}
    for split_file, split_key in [("train.tsv", "train"), ("dev.tsv", "validation"), ("test.tsv", "test")]:
        scripted    = load_split(DATA_DIR_SCRIPTED,    split_file)
        spontaneous = load_split(DATA_DIR_SPONTANEOUS, split_file)

        # Upsample spontaneous on train only — never on validation/test
        """
        if split_key == "train":
            spontaneous = concatenate_datasets([spontaneous] * UPSAMPLE_FACTOR)
            logger.info(
                f"[train] scripted: {len(scripted)} samples | "
                f"spontaneous: {len(spontaneous)} samples after x{UPSAMPLE_FACTOR} upsampling"
            )
        """
        splits[split_key] = concatenate_datasets([scripted, spontaneous]).shuffle(seed=42)

    dataset = DatasetDict(splits)
    dataset = dataset.cast_column("path", Audio(sampling_rate=SAMPLING_RATE))
    dataset = dataset.rename_column("path", "audio")

    logger.info(f"Dataset loaded: {dataset}")
    return dataset

# ─────────────────────────────────────────────
# 4. Preprocessing: audio → log-mel + text → token ids
# ─────────────────────────────────────────────

def make_prepare_fn(feature_extractor, tokenizer):
    def prepare_dataset(batch):
        audio = batch["audio"]

        # Truncate / pad audio to MAX_AUDIO_LEN_S
        samples = audio["array"]
        sr      = audio["sampling_rate"]
        max_samples = int(MAX_AUDIO_LEN_S * sr)
        if len(samples) > max_samples:
            samples = samples[:max_samples]

        # Compute log-mel spectrogram
        batch["input_features"] = feature_extractor(
            samples, sampling_rate=sr, return_tensors="np"
        ).input_features[0]

        # Tokenize transcription
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    return prepare_dataset


# ─────────────────────────────────────────────
# 5. Data collator with dynamic padding
# ─────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # --- input features (already fixed-length mel spectrograms) ---
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt", return_attention_mask=True
        )

        # --- labels: pad to longest sequence in batch ---
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 so it's ignored in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Strip BOS token prepended by the tokenizer if present
        if (
            labels[:, 0] == self.decoder_start_token_id
        ).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ─────────────────────────────────────────────
# 6. Metric: Word Error Rate
# ─────────────────────────────────────────────

def make_compute_metrics(tokenizer):
    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")
    
    def compute_metrics(pred):
        pred_ids    = pred.predictions
        label_ids   = pred.label_ids

        # Replace -100 back to pad token id before decoding
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
        cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    return compute_metrics

def save_test_transcriptions(trainer, dataset, tokenizer, output_dir):
    """Run inference on the test set and save predictions alongside references."""
    predictions = trainer.predict(dataset)
    
    pred_ids  = predictions.predictions
    label_ids = predictions.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    rows = [
        {"reference": ref, "hypothesis": hyp, "wer": jiwer.wer(ref, hyp) * 100}
        for ref, hyp in zip(label_str, pred_str)
    ]

    # Sort by descending WER so worst examples are at the top
    rows = sorted(rows, key=lambda x: x["wer"], reverse=True)

    output_path = os.path.join(output_dir, "test_transcriptions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    logger.info(f"Transcriptions saved to {output_path}")
    return rows

# ─────────────────────────────────────────────
# 7. Main training routine
# ─────────────────────────────────────────────

def main():
    # 7a. Processor
    feature_extractor, tokenizer, processor = build_processor()

    # 7b. Dataset
    raw_datasets = load_local_dataset()

    # 7c. Preprocessing (map in parallel)
    prepare_fn = make_prepare_fn(feature_extractor, tokenizer)
    vectorized_datasets = raw_datasets.map(
        prepare_fn,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=4,      # adjust to your CPU count
        desc="Preprocessing audio",
    )

    # 7d. Model
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.generation_config.language = LANGUAGE
    model.generation_config.task     = TASK
    model.generation_config.forced_decoder_ids = None   # let the tokenizer handle it
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []

    # 7e. Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # 7f. Metrics
    compute_metrics = make_compute_metrics(tokenizer)

    # 7g. Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,

        # Batching
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACC,

        # Optimizer & scheduler
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        lr_scheduler_type="cosine",

        # Precision
        fp16=FP16,
        bf16=BF16,
        
        weight_decay=0.05,
        
        # Memory optimizations
        #gradient_checkpointing=True,   # trades compute for ~40% less VRAM
        gradient_checkpointing=False,  # was True
        optim="adamw_torch_fused",     # faster fused AdamW

        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST,
        greater_is_better=False,

        # Generation at eval time
        predict_with_generate=True,
        generation_max_length=225,

        # Logging
        logging_steps=LOGGING_STEPS,
        report_to=["tensorboard"],

        # Misc
        push_to_hub=False,
        dataloader_num_workers=4,
    )

    # 7h. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["validation"],
        tokenizer=processor.feature_extractor,   # used for saving
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
    )

    # 7i. Train
    logger.info("Starting training …")
    trainer.train()

    # 7j. Save final model + processor
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model saved to {OUTPUT_DIR}")

    # 7k. Final evaluation
    logger.info("Running final evaluation …")
    results = trainer.evaluate()
    logger.info(f"Final WER: {results['eval_wer']:.2f}%")
    logger.info(f"Final CER: {results['eval_cer']:.2f}%")

    with open(os.path.join(OUTPUT_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    

    # 7l. Evaluation + transcriptions on test set
    logger.info("Running evaluation on test set …")
    test_results = trainer.evaluate(eval_dataset=vectorized_datasets["test"])
    logger.info(f"Test WER: {test_results['eval_wer']:.2f}%")

    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    rows = save_test_transcriptions(
        trainer, vectorized_datasets["test"], tokenizer, OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
