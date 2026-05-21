"""
Fine-tuning facebook/wav2vec2-base for ASR with CTC
Hardware target : single NVIDIA L40S (48 GB VRAM)
Data            : ~30h scripted + ~35h spontaneous (Common Voice format)

Requirements:
    pip install transformers>=4.36 datasets accelerate soundfile librosa \
                evaluate jiwer tensorboard torch torchaudio pandas
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import math

import torch
import numpy as np
import pandas as pd
import jiwer
import evaluate
from datasets import load_dataset, Audio, DatasetDict, Dataset, concatenate_datasets
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────

MODEL_ID             = "facebook/wav2vec2-base"
OUTPUT_DIR           = "./wav2vec2-base-finetuned"
VOCAB_DIR            = "./vocab"                   # where vocab.json will be saved
DATA_DIR_SCRIPTED    = "./data/scripted"
DATA_DIR_SPONTANEOUS = "./data/spontaneous"
UPSAMPLE_FACTOR      = 1
SAMPLING_RATE        = 16_000
MAX_AUDIO_LEN_S      = 20                          # wav2vec2 works best under 20 s


# Training hyperparameters, tuned for 65 h on an L40S
TRAIN_BATCH_SIZE  = 16
EVAL_BATCH_SIZE   = 8
GRADIENT_ACC      = 2     # effective batch = 32
LEARNING_RATE     = 1e-4  # CTC fine-tuning typically uses a higher LR than seq2seq
WARMUP_STEPS      = 1_800
MAX_STEPS         = 18_000
SAVE_STEPS        = 500
EVAL_STEPS        = 500
LOGGING_STEPS     = 25
WEIGHT_DECAY      = 0.05
ATTENTION_DROPOUT = 0.05
HIDDEN_DROPOUT    = 0.05
FEAT_PROJ_DROPOUT = 0.05  # dropout on the feature projection layer (wav2vec2-specific)
MASK_TIME_PROB    = 0.075 # SpecAugment-equivalent: proportion of time steps masked
MASK_FEATURE_PROB = 0.004 # proportion of feature channels masked


# ─────────────────────────────────────────────
# 2. Dataset loading
# ─────────────────────────────────────────────

def load_split(data_dir: str, tsv_filename: str) -> Dataset:
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
    splits = {}
    for tsv_file, split_key in [
        ("train.tsv", "train"),
        ("dev.tsv",   "validation"),
        ("test.tsv",  "test"),
    ]:
        scripted    = load_split(DATA_DIR_SCRIPTED,    tsv_file)
        spontaneous = load_split(DATA_DIR_SPONTANEOUS, tsv_file)

        if split_key == "train" and UPSAMPLE_FACTOR > 1:
            scripted = concatenate_datasets([scripted] * UPSAMPLE_FACTOR)
            logger.info(
                f"[train] scripted: {len(scripted)} | "
                f"spontaneous: {len(spontaneous)} (no upsampling needed)"
            )

        splits[split_key] = concatenate_datasets([scripted, spontaneous]).shuffle(seed=42)

    dataset = DatasetDict(splits)
    dataset = dataset.cast_column("path", Audio(sampling_rate=SAMPLING_RATE))
    dataset = dataset.rename_column("path", "audio")
    logger.info(f"Dataset loaded: {dataset}")
    return dataset

def is_valid(batch):
    vals = batch["input_values"]
    return not (any(math.isnan(v) for v in vals) or any(math.isinf(v) for v in vals))

# ─────────────────────────────────────────────
# 3. Vocabulary built from training transcriptions
# ─────────────────────────────────────────────

def build_vocabulary(dataset: Dataset, vocab_dir: str) -> str:
    """Extract character set from training transcriptions and save vocab.json."""
    os.makedirs(vocab_dir, exist_ok=True)

    def extract_chars(batch):
        all_text = " ".join(batch["transcription"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = dataset.map(
        extract_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset.column_names,
    )

    vocab_set = set()
    for v in vocabs["vocab"]:
        vocab_set.update(v)

    # Build vocab dict; reserve special tokens
    vocab_dict = {c: i for i, c in enumerate(sorted(vocab_set))}

    # Replace space with a readable token
    if " " in vocab_dict:
        idx = vocab_dict.pop(" ")
        vocab_dict["|"] = idx   # | is the conventional word-boundary token

    # Add CTC blank and unknown tokens
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)   # PAD doubles as CTC blank

    vocab_path = os.path.join(vocab_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    logger.info(f"Vocabulary of {len(vocab_dict)} tokens saved to {vocab_path}")
    return vocab_path


# ─────────────────────────────────────────────
# 4. Processor (feature extractor + CTC tokenizer)
# ─────────────────────────────────────────────

def build_processor(vocab_dir: str) -> Wav2Vec2Processor:
    tokenizer = Wav2Vec2CTCTokenizer(
        os.path.join(vocab_dir, "vocab.json"),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLING_RATE,
        padding_value=0.0,
        do_normalize=True,   # normalize waveform to zero mean, unit variance
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    return processor


# ─────────────────────────────────────────────
# 5. Preprocessing
#   - Input  : raw waveform
#   - Labels : character token ids
# ─────────────────────────────────────────────

def make_prepare_fn(processor: Wav2Vec2Processor):
    def prepare_dataset(batch):
        audio = batch["audio"]
        samples = audio["array"]
        sr      = audio["sampling_rate"]

        # Truncate to MAX_AUDIO_LEN_S
        max_samples = int(MAX_AUDIO_LEN_S * sr)
        if len(samples) > max_samples:
            samples = samples[:max_samples]
        
        # Skip silent clips — normalisation would produce NaN
        if samples.std() < 1e-7:
            logger.warning("Near-silent audio detected, replacing with noise floor")
            samples = np.random.normal(0, 1e-6, samples.shape).astype(np.float32)

        # Extract raw waveform features (normalised)
        batch["input_values"] = processor(
            samples, sampling_rate=sr
        ).input_values[0]
        batch["input_length"] = len(batch["input_values"])

        # Encode transcription as character ids
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids

        return batch

    return prepare_dataset


# ─────────────────────────────────────────────
# 6. Data collator with dynamic padding
#
# CTC requires padding both input_values and labels,
# and the two sequences can have very different lengths.
# ─────────────────────────────────────────────

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Pad input waveforms
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Pad labels separately (different length than input)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")

        # Replace padding id with -100 so CTC loss ignores it
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# ─────────────────────────────────────────────
# 7. Metric: WER
# ─────────────────────────────────────────────

def make_compute_metrics(processor: Wav2Vec2Processor):
    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")
    
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids    = np.argmax(pred_logits, axis=-1)
        label_ids   = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
        cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}

    return compute_metrics


# ─────────────────────────────────────────────
# 8. Save test transcriptions
# ─────────────────────────────────────────────

def save_test_transcriptions(trainer, dataset, processor, output_dir):
    predictions = trainer.predict(dataset)

    pred_ids  = np.argmax(predictions.predictions, axis=-1)
    label_ids = predictions.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    rows = [
        {"reference": ref, "hypothesis": hyp, "wer": jiwer.wer(ref, hyp) * 100, "cer": jiwer.cer(ref, hyp) * 100}
        for ref, hyp in zip(label_str, pred_str)
    ]
    rows = sorted(rows, key=lambda x: x["wer"], reverse=True)

    output_path = os.path.join(output_dir, "test_transcriptions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    logger.info(f"Transcriptions saved to {output_path}")
    return rows


# ─────────────────────────────────────────────
# 9. Main
# ─────────────────────────────────────────────

def main():
    # 9a. Dataset
    raw_datasets = load_local_dataset()

    # 9b. Vocabulary (built from training data only)
    build_vocabulary(raw_datasets["train"], VOCAB_DIR)

    # 9c. Processor
    processor = build_processor(VOCAB_DIR)

    # 9d. Preprocessing
    prepare_fn = make_prepare_fn(processor)
    vectorized_datasets = raw_datasets.map(
        prepare_fn,
        remove_columns=raw_datasets["train"].column_names,
        num_proc=4,
        desc="Preprocessing audio",
    )

    # Filter out samples that are too long for the GPU (optional safety net)
    max_input_length = MAX_AUDIO_LEN_S * SAMPLING_RATE
    vectorized_datasets = vectorized_datasets.filter(
        lambda x: x < max_input_length,
        input_columns=["input_length"],
    )
    vectorized_datasets = vectorized_datasets.filter(is_valid, num_proc=4)
    
    # Sanity check: decode a few training labels
    sample = vectorized_datasets["train"][0]
    label_ids = [l for l in sample["labels"] if l != -100]
    decoded = processor.tokenizer.decode(label_ids)
    print(f"Decoded label: '{decoded}'")
    print(f"Original transcription: '{raw_datasets['train'][0]['transcription']}'")

    # 9e. Model
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        attention_dropout=ATTENTION_DROPOUT,
        hidden_dropout=HIDDEN_DROPOUT,
        feat_proj_dropout=FEAT_PROJ_DROPOUT,
        mask_time_prob=MASK_TIME_PROB,
        mask_feature_prob=MASK_FEATURE_PROB,
        layerdrop=0.0,
        ctc_loss_reduction="mean",       # "mean" is more stable than "sum" for variable lengths
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,    # needed because we replace the LM head with our vocab
    )
    logger.info(f"LM head output size : {model.lm_head.out_features}")
    logger.info(f"Tokenizer vocab size: {len(processor.tokenizer)}")

    # Freeze the feature encoder for the first part of training —
    # the convolutional frontend is already well trained on LibriSpeech;
    # fine-tuning it on 65h risks overfitting and slows training significantly.
    model.freeze_feature_encoder()

    # 9f. Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # 9g. Metrics
    compute_metrics = make_compute_metrics(processor)

    # 9h. Training arguments
    # Note: Trainer (not Seq2SeqTrainer) — CTC is not a seq2seq model
    training_args = TrainingArguments(
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
        weight_decay=WEIGHT_DECAY,
        max_grad_norm = 0.5,   # default is 1.0; 0.5 is safer for CTC

        # Precision — gradient checkpointing is safe with CTC (no seq2seq conflict)
        fp16=True,
        gradient_checkpointing=True,

        # Evaluation & saving
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        # Logging
        logging_steps=LOGGING_STEPS,
        report_to=["tensorboard"],

        # Misc
        push_to_hub=False,
        dataloader_num_workers=4,
        group_by_length=True,   # batch samples of similar length together
                                # → reduces padding waste, speeds up training significantly
    )

    # 9i. Trainer (standard, not Seq2Seq)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=vectorized_datasets["train"],
        eval_dataset=vectorized_datasets["validation"],
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
    )

    # 9j. Train
    logger.info("Starting training …")
    trainer.train()

    # 9k. Save
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    logger.info(f"Model saved to {OUTPUT_DIR}")

    # 9l. Test set evaluation
    logger.info("Running evaluation on test set …")
    test_results = trainer.evaluate(eval_dataset=vectorized_datasets["test"])
    logger.info(f"Test WER: {test_results['eval_wer']:.2f}%")
    logger.info(f"Test CER: {test_results['eval_cer']:.2f}%")

    with open(os.path.join(OUTPUT_DIR, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    save_test_transcriptions(trainer, vectorized_datasets["test"], processor, OUTPUT_DIR)


if __name__ == "__main__":
    main()
