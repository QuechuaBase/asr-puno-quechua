"""
Prepare Collao corpus for CPT.

Extracts Audios/ from data/collao/Quechua_Collao_Corpus.zip,
resamples all WAVs to 16kHz mono, and builds a fairseq pretrain
manifest at data/manifests/pretrain/collao.tsv.

Usage (from project root):
    conda activate asr-puno
    python data/prepare_collao_cpt.py
"""

import zipfile
from pathlib import Path

import soundfile as sf
import torchaudio

ROOT        = Path(__file__).resolve().parent.parent
ZIP_PATH    = ROOT / "data" / "collao" / "Quechua_Collao_Corpus.zip"
WAV_DIR     = ROOT / "data" / "collao" / "wav"
MANIFEST    = ROOT / "data" / "manifests" / "pretrain" / "collao.tsv"
DOCKER_ROOT = "/workspace/data/collao/wav"
TARGET_SR   = 16000
MIN_FRAMES  = 16000       # 1s
MAX_FRAMES  = 240000      # 15s


def main():
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Extract ---
    print("Extracting WAVs from zip...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        wav_entries = [e for e in zf.namelist()
                       if e.startswith("Audios/") and e.endswith(".wav")]
        already = sum(1 for e in wav_entries
                      if (WAV_DIR / Path(e).name).exists())
        print(f"  {len(wav_entries)} WAVs in zip, {already} already extracted")
        for entry in wav_entries:
            out_path = WAV_DIR / Path(entry).name
            if out_path.exists():
                continue
            with zf.open(entry) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
    print(f"  Done. WAVs in {WAV_DIR.relative_to(ROOT)}")

    # --- Step 2: Resample + build manifest ---
    print("Resampling to 16kHz mono and building manifest...")
    wav_files = sorted(WAV_DIR.glob("*.wav"))
    print(f"  {len(wav_files)} files to process")

    entries = []
    skipped_short = skipped_long = skipped_err = 0

    for i, wav_path in enumerate(wav_files):
        if i % 1000 == 0:
            print(f"  {i}/{len(wav_files)}...")
        try:
            info = sf.info(str(wav_path))
            # Resample if needed
            if info.samplerate != TARGET_SR or info.channels > 1:
                waveform, sr = torchaudio.load(str(wav_path))
                if sr != TARGET_SR:
                    waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                torchaudio.save(str(wav_path), waveform, TARGET_SR)
                num_frames = waveform.shape[-1]
            else:
                num_frames = info.frames
        except Exception as e:
            print(f"  WARNING: {wav_path.name}: {e}")
            skipped_err += 1
            continue

        if num_frames < MIN_FRAMES:
            skipped_short += 1
            continue
        if num_frames > MAX_FRAMES:
            skipped_long += 1
            continue

        entries.append((wav_path.name, num_frames))

    print(f"  Kept: {len(entries)}  |  skipped short: {skipped_short}  "
          f"long: {skipped_long}  error: {skipped_err}")

    # --- Step 3: Write manifest ---
    with open(MANIFEST, "w") as f:
        f.write(DOCKER_ROOT + "\n")
        for fname, nframes in entries:
            f.write(f"{fname}\t{nframes}\n")

    total_s = sum(n for _, n in entries) / TARGET_SR
    print(f"Wrote {len(entries)} entries → {MANIFEST.relative_to(ROOT)}")
    print(f"Total duration: {total_s:.0f}s = {total_s/3600:.1f}h")


if __name__ == "__main__":
    main()
