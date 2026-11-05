# asr-puno-quechua

Automatic speech recognition for Puno Quechua (qxp) using Mozilla Common Voice data. We fine-tune wav2vec 2.0 models with continued pre-training (CPT) on Puno Quechua audio and evaluate on scripted and spontaneous speech.

## Results

Word and character error rates (%) on the test partitions of the Puno Quechua corpus, as reported in the paper (Table 2). Also in [`results/paper_table2.csv`](results/paper_table2.csv).

| Model | Params | SCS-26 WER | SCS-26 CER | SPS-4 WER | SPS-4 CER | OOD WER | OOD CER | Mean WER | Mean CER |
|-------|-------:|-----------:|-----------:|----------:|----------:|--------:|--------:|---------:|---------:|
| xls-r-300m (fine-tuned) | 315M | 16.5 | 3.2 | 16.7 | 2.9 | 28.4 | **4.3** | 20.5 | 3.5 |
| xls-r-300m + CPT (fine-tuned) | 315M | **11.6** | **2.5** | **16.0** | **2.8** | 26.9 | 4.5 | **18.2** | **3.3** |
| omniASR_CTC_300M | 325M | 43.3 | 8.1 | 33.3 | 5.6 | 41.0 | 6.0 | 39.2 | 6.6 |
| omniASR_LLM_300M | 1.6B | 22.4 | 4.0 | 20.5 | 3.7 | **24.4** | 3.7 | 22.4 | 3.8 |

**SCS-26** = Common Voice Scripted Speech v26; **SPS-4** = Common Voice Spontaneous Speech v4; **OOD** = 0.27 h out-of-domain set from radio and social media, used for evaluation only. **CPT** = continued pre-training on the unlabelled corpus audio before fine-tuning; omniASR models are zero-shot baselines. Both datasets were split 80/10/10 (train/dev/test), and the fine-tuned models also use "silver" transcriptions of unvalidated speech generated with omniASR_LLM_7B.

Models on HuggingFace: [xls-r-qxp-cv26-silver](https://huggingface.co/QuechuaBase/xls-r-qxp-cv26-silver), [xls-r-cpt-qxp-cv26-silver](https://huggingface.co/QuechuaBase/xls-r-cpt-qxp-cv26-silver).

> The numbers from our first evaluation (different data version and splits) are archived in [`results/first_evaluation/`](results/first_evaluation/README.md) for reference. Do not compare them directly with the table above.

## Run inference

See [`inference/README.md`](inference/README.md) — clone the repo, download the checkpoint from HuggingFace, and transcribe audio with one Docker command. No conda or fairseq install needed.

## Datasets

Both datasets are from the [Mozilla Data Collective](https://datacollective.mozillafoundation.org/) and cover Puno Quechua (ISO 639-3: qxp). Licensed CC0-1.0.

| | Scripted Speech 26.0 (SCS-26) | Spontaneous Speech 4.0 (SPS-4) |
|---|---|---|
| **Speech type** | Speakers read pre-written sentences | Speakers respond naturally to prompts |
| **Clips** | 25,423 | 7,305 |
| **Recorded duration** | 34.88 h | 35.41 h |
| **Validated duration** | 31.18 h | 5.18 h |
| **Avg. clip duration** | 4.94 s | 17.45 s |
| **Speakers** | 81 | 110 |
| **Prompts** | 2,070 sentences | 150 questions |

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
