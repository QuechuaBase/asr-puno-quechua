# asr-puno-quechua

**Automatic speech recognition (ASR) for Puno Quechua (ISO 639-3: `qxp`), trained and evaluated on a community-collected Common Voice corpus.**

Puno Quechua is a Collao Quechua variety spoken by about 214,000 people in the Puno region of Peru. Few of them read and write it, so speech interfaces are a practical route to digital tools in the language. This repository documents one experiment: **how well a multilingual self-supervised speech model (xls-r-300m) recognises Puno Quechua after fine-tuning on about 30 hours of community-recorded speech, and whether continued pre-training (CPT) on the corpus audio helps.**

- **Data:** Common Voice Scripted Speech v26 (read speech) and Spontaneous Speech v4 (free answers to questions), both CC0-1.0, plus a 0.27 h out-of-domain (OOD) test set from radio and social media.
- **Models:** xls-r-300m fine-tuned with a character-level CTC objective, with and without CPT, compared with the zero-shot omniASR_CTC_300M and omniASR_LLM_300M models.
- **Main result:** CPT followed by fine-tuning gives the lowest error on in-domain speech: **11.6 % WER / 2.5 % CER** on read speech and **16.0 % / 2.8 %** on spontaneous speech. On the OOD set, the zero-shot omniASR_LLM_300M has the lowest WER (24.4 %).
- **Released:** data splits, silver transcriptions, training and evaluation code, and two fine-tuned checkpoints on HuggingFace.

---

## Contents

1. [Data](#data)
2. [Method](#method)
3. [Results](#results)
4. [Models and inference](#models-and-inference)
5. [Reproducing the experiment](#reproducing-the-experiment)
6. [Repository layout](#repository-layout)
7. [Licence](#licence)

---

## Data

Full documentation, including release history, split statistics is in **[`data/README.md`](data/README.md)**.

### Source corpus

The corpus was recorded between January 2025 and March 2026 by 151 native speakers on the localised [Common Voice platform](https://commonvoice.mozilla.org/qxp). It was collected through a participatory design process with the National University of Altiplano of Puno, the Puno Regional Government and the community organisation Illariy Ch'aska, and community members validated the clips. It is released on the Mozilla Data Collective under **CC0-1.0**.

| | Scripted Speech 26.0 (SCS-26) | Spontaneous Speech 4.0 (SPS-4) |
|---|---|---|
| **Speech type** | Speakers read seed sentences | Speakers answer open-ended questions |
| **Clips** | 25,423 | 7,305 |
| **Recorded duration** | 34.88 h | 35.41 h |
| **Validated duration** | 31.18 h | 5.18 h |
| **Avg. clip duration** | 4.94 s | 17.45 s |
| **Speakers** | 81 | 110 |
| **Prompts** | 2,070 sentences | 150 questions |

**Versions.** Common Voice publishes a new cumulative release every quarter. This experiment uses **Scripted Speech 26.0** and **Spontaneous Speech 4.0** (data cut-off 12 June 2026). The later releases 27.0 / 5.0 contain the same Puno Quechua clips. All releases are listed at **<https://commonvoice.mozilla.org/en/datasets>**.

### Splits used in this experiment ([`data/splits_cv26/`](data/splits_cv26/))

The splits are speaker-disjoint and gender-stratified, with a target ratio of 80/10/10 by duration. A cap on any single speaker's share of dev and test keeps those sets from being dominated by a few speakers, so the realised splits favour train:

| | Train | Dev | Test |
|---|---|---|---|
| **Scripted** (curated transcripts) | 20,870 clips · 27.80 h · 35 spk | 948 · 1.35 h · 21 spk | 909 · 1.35 h · 21 spk |
| **Spontaneous** (gold + silver) | 5,902 · 26.17 h · 64 spk | 575 · 1.87 h · 24 spk | 546 · 1.88 h · 22 spk |

- **Curated** transcripts are the validated scripted sentences, lowercased, with punctuation removed and Puno Quechua spelling applied.
- **Gold** transcripts are community-validated human transcriptions of spontaneous clips.
- **Silver** transcripts were generated automatically with **omniASR_LLM_7B** for spontaneous clips that had no validated transcription. They provide 20 h of the 26 h of spontaneous training data.
- **OOD** ([`data/additional_data/`](data/additional_data/)) contains 109 clips (0.27 h) from radio drama, interviews and social media, with manual transcriptions. It is used only for evaluation.

---

## Method

### Overview

```
XLS-R 300M (128 languages, self-supervised)
   │
   ├──────────────► CTC fine-tuning ──► xls-r-300m (fine-tuned)
   │
   └─► CPT on corpus audio ─► CTC fine-tuning ──► xls-r-300m + CPT (fine-tuned)

omniASR_CTC_300M / omniASR_LLM_300M ──► zero-shot (no training on this corpus)
```

### 1. Audio and text preprocessing

- **Audio:** MP3 converted to 16 kHz mono 16-bit WAV with `ffmpeg`. Fine-tuning skips clips longer than 30 s.
- **Text:** lowercased, with `? ! ¿ ¡ . ,` removed. Scripted references use the curated spelling. Spontaneous gold references have disfluency markers (e.g. `[ aaa]`) removed.
- **Output units:** characters (a character-level CTC vocabulary), with `|` as the word boundary.

### 2. Continued pre-training (CPT)

CPT adapts XLS-R to Puno Quechua acoustics before any transcripts are used. It continues training with the original wav2vec 2.0 self-supervised objective (contrastive InfoNCE loss with quantised targets).

| Setting | Value |
|---|---|
| Initial model | XLS-R 300M (`xlsr2_300m.pt`, 128 languages) |
| Audio | Unlabelled scripted and spontaneous corpus audio (about 66 h), mixed equally, crops of 1–15 s |
| Updates | 10,000 (1,000 warm-up, then polynomial decay) |
| Peak learning rate | 1 × 10⁻⁴ (Adam, β = 0.9/0.98, weight decay 0.01) |
| Effective batch | about 25 min of audio per update (1.5 M samples × 16 accumulation steps) |
| Precision | fp32 |

Config: [`training/configs/w2v2-large-cpt_qxp.yaml`](training/configs/w2v2-large-cpt_qxp.yaml). The CPT audio includes clips from the dev and test splits. Only audio is used, never transcripts.

### 3. Supervised fine-tuning

Both models (with and without CPT) are fine-tuned the same way, on the **train** split of both datasets: curated scripted plus spontaneous gold and silver. That training set is what the "-silver" in the checkpoint names refers to.

| Setting | Value |
|---|---|
| Objective | CTC over characters |
| Updates | 20,000. The transformer is frozen for the first 10,000 updates, during which only the output layer is trained. |
| Learning rate | 5 × 10⁻⁵, tri-stage schedule (10 % warm-up, 65 % constant, 25 % decay) |
| Regularisation | SpecAugment-style masking (time 0.75, channel 0.25 / length 64), layer drop 0.1 |
| Effective batch | about 12 min of audio per update (5.6 M samples × 2 accumulation steps) |
| Model selection | Checkpoint with the lowest WER on dev |
| Precision | fp16 |

Config: [`training/configs/w2v2-large-finetune_qxp.yaml`](training/configs/w2v2-large-finetune_qxp.yaml). Gradient accumulation is scaled with the number of GPUs, so the effective batch size stays the same. Training was run with fairseq in Docker on NVIDIA L40S GPUs.

### 4. Baselines

**omniASR_CTC_300M** and **omniASR_LLM_300M** (Omnilingual ASR, which includes Puno Quechua) are run **zero-shot** with language code `qxp_Latn`, without training on this corpus. Code: [`eval/omnilingual/`](eval/omnilingual/).

### 5. Decoding and scoring

- **Decoding:** greedy (Viterbi) CTC, with **no language model**.
- **Scoring:** hypotheses and references are normalised the same way (lowercased, punctuation removed). WER and CER are computed with [`jiwer`](https://github.com/jitsi/jiwer) as **corpus-level** rates (total edits divided by total reference words or characters).
- **Evaluation sets:** SCS-26 test, SPS-4 test and OOD, each scored separately. "Mean" is the unweighted mean of the three.
- **Code:** [`eval/fairseq/`](eval/fairseq/) (fine-tuned models) and [`eval/omnilingual/compute_wer.py`](eval/omnilingual/compute_wer.py) (scoring).

---

## Results

Word and character error rates (%) on the test sets.

| Model | Params | SCS-26 WER | SCS-26 CER | SPS-4 WER | SPS-4 CER | OOD WER | OOD CER | Mean WER | Mean CER |
|-------|-------:|-----------:|-----------:|----------:|----------:|--------:|--------:|---------:|---------:|
| xls-r-300m (fine-tuned) | 315M | 16.5 | 3.2 | 16.7 | 2.9 | 28.4 | **4.3** | 20.5 | 3.5 |
| xls-r-300m + CPT (fine-tuned) | 315M | **11.6** | **2.5** | **16.0** | **2.8** | 26.9 | 4.5 | **18.2** | **3.3** |
| omniASR_CTC_300M (zero-shot) | 325M | 43.3 | 8.1 | 33.3 | 5.6 | 41.0 | 6.0 | 39.2 | 6.6 |
| omniASR_LLM_300M (zero-shot) | 1.6B | 22.4 | 4.0 | 20.5 | 3.7 | **24.4** | 3.7 | 22.4 | 3.8 |

**SCS-26** = Scripted Speech v26 test set; **SPS-4** = Spontaneous Speech v4 test set; **OOD** = out-of-domain set. See [Data](#data) and [Method](#method) for definitions.

Details are in [`data/README.md`](data/README.md#53-known-limitations-of-the-splits).

> An earlier evaluation round used different data releases and splits. Its numbers are archived in [`results/first_evaluation/`](results/first_evaluation/README.md) and **cannot be compared** with the table above.

---

## Models and inference

Checkpoints: [xls-r-qxp-cv26-silver](https://huggingface.co/QuechuaBase/xls-r-qxp-cv26-silver) (no CPT) and [xls-r-cpt-qxp-cv26-silver](https://huggingface.co/QuechuaBase/xls-r-cpt-qxp-cv26-silver) (with CPT).

See [`inference/README.md`](inference/README.md): clone the repo, download a checkpoint from HuggingFace, and transcribe audio with one Docker command. No conda or fairseq install is needed.

---

## Reproducing the experiment

### 1. Setup

```bash
conda create -n asr-puno python=3.10
conda activate asr-puno
pip install -r requirements.txt
```

### 2. Get the audio

The Mozilla Data Collective agreement forbids redistribution, so this repository contains **no Common Voice audio**. Create an account at the [Mozilla Data Collective](https://mozilladatacollective.com), accept the data usage agreement, and download the two releases. You can do this from the web page or with the script:

```bash
cp .env.example .env          # add your MDC_API_KEY (Profile > API)
python data/download_data.py  # saves to data/scripted/ and data/spontaneous/
```

> `download_data.py` still points to older releases. To reproduce this experiment, download **latest scripted and spontaneous released datasets**: [`data/README.md`](data/README.md#2-release-history-on-the-mozilla-data-collective).

### 3. Continued pre-training and fine-tuning

Training requires Docker with GPU support. See [`training/README.md`](training/README.md) and [`pipeline.sh`](pipeline.sh) for the full pipeline: continued pre-training, then fine-tuning, then evaluation.

### 4. Evaluation

Evaluation scripts are in [`eval/fairseq/`](eval/fairseq/) (fine-tuned models) and [`eval/omnilingual/`](eval/omnilingual/) (zero-shot baselines and scoring). See [Method § 5](#5-decoding-and-scoring) for the decoding and scoring settings.

---

## Repository layout

```
data/          Splits, curated and silver transcripts, OOD set (see data/README.md)
training/      CPT and fine-tuning configs and launch scripts (fairseq, Docker)
eval/          Evaluation of fine-tuned models (fairseq) and omniASR baselines
inference/     Transcribe audio with the released checkpoints (Docker)
results/       Results table and archived first-evaluation outputs
pipeline.sh    End-to-end pipeline: CPT → fine-tuning → evaluation
```

---

## Licence

- **Code:** Apache-2.0 (see [`LICENSE`](LICENSE)).
- **Common Voice data and the derived splits and transcripts:** CC0-1.0.
