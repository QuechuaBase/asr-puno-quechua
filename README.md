# asr-puno-quechua

Automatic speech recognition for Puno Quechua (qxp) using Mozilla Common Voice data. We fine-tune wav2vec 2.0 models with continued pre-training (CPT) on Puno Quechua audio and evaluate on scripted and spontaneous speech.

## Results

| Model | Scripted WER | Scripted CER | Spontaneous WER | Spontaneous CER |
|-------|-------------:|-------------:|----------------:|----------------:|
| ft_cpt_validated | **1.19%** | 0.19% | 13.61% | 1.73% |
| ft_cpt_silver | 2.11% | 0.30% | **3.15%** | **0.41%** |
| ft_xlsr_validated | 2.06% | 0.30% | 13.58% | 1.71% |
| ft_xlsr_silver | 4.36% | 0.57% | 6.68% | 0.81% |
| omniASR LLM 300M v2 | 7.07% | 0.89% | 11.98% | 1.54% |
| omniASR CTC 300M v2 | 24.47% | 3.05% | 20.22% | 2.43% |

**CPT** = our continued pre-trained checkpoint; **XLSR** = public XLSR-128 300M baseline; **omniASR** = zero-shot Omnilingual baseline.  
**Validated** = trained on gold-labeled data only; **Silver** = trained with additional auto-transcribed spontaneous speech.

Silver data trades ~1–2% scripted WER for ~10% spontaneous WER — worth it if your use case involves conversational speech.

## Run inference

See [`inference/README.md`](inference/README.md) — clone the repo, download the checkpoint from HuggingFace, and transcribe audio with one Docker command. No conda or fairseq install needed.

## Datasets

Both datasets are from the [Mozilla Data Collective](https://datacollective.mozillafoundation.org/) and cover Puno Quechua (ISO 639-3: qxp). Licensed CC0-1.0.

| | Scripted Speech 25.0 | Spontaneous Speech 3.0 |
|---|---|---|
| **Speech type** | Speakers read pre-written sentences | Speakers respond naturally to prompts |
| **Clips** | 25,382 | 7,286 |
| **Validated** | 22,727 (89.5%) | 1,074 (14.7%) |
| **Validated duration** | 31.2 hours | 5.2 hours |
| **Avg. clip duration** | 4.937s | 17.45s |
| **Speakers** | 81 | 110 |

## Setup (for training / evaluation)

```bash
conda create -n asr-puno python=3.10
conda activate asr-puno
pip install -r requirements.txt
```

## Download data

Get an API key from your profile at [datacollective.mozillafoundation.org](https://datacollective.mozillafoundation.org/) (Profile > API), then:

```bash
cp .env.example .env
# fill in your key in .env
python data/download_data.py
```

Data is saved to `data/scripted/` and `data/spontaneous/`.

## Training

Model training requires Docker with GPU support. See [`pipeline.sh`](pipeline.sh) for the full pipeline: continued pre-training → fine-tuning → evaluation.
