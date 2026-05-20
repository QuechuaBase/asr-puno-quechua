"""
Prepare data/additional_data/ for evaluation against trained models.

Creates:
  data/additional_data.tsv              <- for run_omnilingual.py (path + sentence)
  data/additional_data_16k/*.wav        <- resampled to 16kHz
  data/manifests/additional/
      test.tsv                          <- fairseq manifest (Docker paths)
      test.ltr                          <- reference transcriptions in LTR format
      dict.ltr.txt                      <- copied from --dict_path

Usage (from project root):
  conda activate asr-puno
  python eval/omnilingual/prepare_additional_data.py
  python eval/omnilingual/prepare_additional_data.py \\
      --dict_path data/manifests/finetune/qxp_v2/dict.ltr.txt
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import soundfile as sf
import torchaudio

ROOT           = Path(__file__).resolve().parents[2]
ADDITIONAL_DIR = ROOT / "data" / "additional_data"
RESAMPLED_DIR  = ROOT / "data" / "additional_data_16k"
OUT_TSV        = ROOT / "data" / "additional_data.tsv"
MANIFEST_DIR   = ROOT / "data" / "manifests" / "additional"
TARGET_SR      = 16000
DOCKER_WAV_ROOT = "/workspace/data/additional_data_16k"

DEFAULT_DICT = ROOT / "data" / "manifests" / "finetune" / "qxp_v2" / "dict.ltr.txt"


def normalize(text: str) -> str:
    """Lowercase and strip punctuation, matching build_finetune_manifests.py."""
    return "".join(c for c in text.lower() if c not in "?!¿¡.,").strip()


def to_ltr(text: str) -> str:
    """Convert normalized sentence to fairseq LTR format (space → '|')."""
    return " ".join("|" if c == " " else c for c in text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dict_path",
        type=Path,
        default=DEFAULT_DICT,
        help="Path to dict.ltr.txt (default: data/manifests/finetune/qxp_v2/dict.ltr.txt)",
    )
    args = parser.parse_args()

    RESAMPLED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(ADDITIONAL_DIR.glob("*.wav"))
    pairs, missing_txt = [], []
    for wav in wav_files:
        txt = wav.with_suffix(".txt")
        if txt.exists():
            pairs.append((wav, txt))
        else:
            missing_txt.append(wav.name)

    if missing_txt:
        print(f"Warning: {len(missing_txt)} WAV files with no matching .txt: "
              f"{missing_txt[:3]}{'...' if len(missing_txt) > 3 else ''}")

    print(f"Found {len(pairs)} paired WAV + TXT files")

    rows, manifest_entries, ltr_lines = [], [], []
    for wav_path, txt_path in pairs:
        sentence = txt_path.read_text(encoding="utf-8").strip()
        normalized = normalize(sentence)

        out_wav = RESAMPLED_DIR / wav_path.name
        if not out_wav.exists():
            waveform, sr = torchaudio.load(str(wav_path))
            if sr != TARGET_SR:
                resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            torchaudio.save(str(out_wav), waveform, TARGET_SR)

        num_frames = sf.info(str(out_wav)).frames
        rows.append({"path": str(wav_path), "sentence": sentence})
        manifest_entries.append((wav_path.name, num_frames))
        ltr_lines.append(to_ltr(normalized))

    # Omnilingual TSV (uses original paths)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_TSV, sep="\t", index=False)
    print(f"Wrote {len(df)} rows → {OUT_TSV.relative_to(ROOT)}")

    # Fairseq manifest (Docker paths)
    manifest_tsv = MANIFEST_DIR / "test.tsv"
    with open(manifest_tsv, "w") as f:
        f.write(DOCKER_WAV_ROOT + "\n")
        for fname, nframes in manifest_entries:
            f.write(f"{fname}\t{nframes}\n")
    print(f"Wrote fairseq manifest → {manifest_tsv.relative_to(ROOT)}")

    # LTR reference file
    ltr_path = MANIFEST_DIR / "test.ltr"
    with open(ltr_path, "w") as f:
        f.write("\n".join(ltr_lines) + "\n")
    print(f"Wrote LTR references → {ltr_path.relative_to(ROOT)}")

    # Dict
    if args.dict_path.exists():
        shutil.copy(args.dict_path, MANIFEST_DIR / "dict.ltr.txt")
        print(f"Copied dict.ltr.txt from {args.dict_path.relative_to(ROOT)}")
    else:
        print(f"Warning: dict not found at {args.dict_path} — copy manually before running fairseq eval")


if __name__ == "__main__":
    main()
