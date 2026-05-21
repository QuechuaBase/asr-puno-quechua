# Puno Quechua ASR — Inference Setup

Transcribe Puno Quechua audio using `ft_cpt_validated` (1.19% WER on scripted speech, 13.61% on spontaneous/conversational).

---

## Requirements

- Docker (with GPU support)
- ~4GB disk for the model checkpoint

---

## Setup

**1. Clone the repo**

```bash
git clone <repo-url>
cd asr-puno-quechua
```

**2. Download the checkpoint**

```bash
pip install huggingface_hub
huggingface-cli download QuechuaBase/xls-r-cpt-qxp-validated \
    checkpoint_best.pt \
    --local-dir checkpoints/ft_cpt_validated/
```

**3. Create your `.env` file**

```bash
cat > .env <<EOF
DOCKER_UID=$(id -u)
DOCKER_GID=$(id -g)
DOCKER_CUDA_VISIBLE_DEVICES=0
EOF
```

This tells Docker to run as your user and use GPU 0. If you have multiple GPUs and want a specific one, change `0` to the relevant index. You do **not** need to run `setup_gpus.sh` — that script is specific to a different machine.

---

## Usage

All commands are run from the project root using `docker compose` (not plain `docker run`). The `docker-compose.yaml` in the repo handles the entrypoint, GPU access, and bind-mounting the project directory — all dependencies are baked into the image.

**Transcribe one or more files**

```bash
docker compose run asr-puno-quechua -c "python inference/transcribe.py recording.wav"
docker compose run asr-puno-quechua -c "python inference/transcribe.py file1.wav file2.wav file3.wav"
```

Output:
```
recording.wav  →  iskay urququnaq chaupinpi payqa tiyan
```

**Transcribe a whole folder (writes TSV)**

```bash
docker compose run asr-puno-quechua -c \
  "python inference/transcribe.py --input_dir ./my_audio/ --output_tsv results.tsv"
```

Output TSV columns: `path`, `transcription`

**Transcribe from a TSV manifest**

If you have a TSV with a `path` column:

```bash
docker compose run asr-puno-quechua -c \
  "python inference/transcribe.py --tsv my_manifest.tsv --output_tsv results.tsv"
```

---

## Reproducing the test set evaluation

The test manifests (`data/manifests/finetune/qxp_v2/test.tsv`, `test_spont.tsv`) are included in the repo, but the audio files are not. To download them:

1. Get an API key from [datacollective.mozillafoundation.org](https://datacollective.mozillafoundation.org) (Profile → API)
2. Add it to `.env`: `MDC_API_KEY=your-key-here`
3. Run: `conda activate asr-puno && python data/download_data.py`

Then transcribe the test set:
```bash
docker compose run asr-puno-quechua -c \
  "python inference/transcribe.py --tsv data/manifests/finetune/qxp_v2/test.tsv --output_tsv results/test_eval.tsv"
```

---

## Notes

- **Audio format**: WAV or MP3, any sample rate, mono or stereo — converted to 16kHz mono automatically
- **GPU**: used automatically if available; falls back to CPU (slower but works)
- **Accuracy**: 1.19% WER on scripted Puno Quechua; 13.61% on spontaneous speech
- **Checkpoint**: `--ckpt PATH` overrides the default checkpoint location if needed
