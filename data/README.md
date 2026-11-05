# Data

This folder holds the data splits, reference transcriptions and evaluation sets used for the automatic speech recognition (ASR) experiments.

No audio from Common Voice is stored in this repository. The Mozilla Data Collective data usage agreement forbids redistribution, so you have to download the audio yourself (see [Obtaining the audio](#6-obtaining-the-audio)). This repository distributes only clip identifiers, split assignments, transcriptions and the out-of-domain (OOD) evaluation set.

---

## 1. Source corpus

The corpus covers **Puno Quechua** (ISO 639-3: `qxp`), a Collao Quechua variety of Southern Quechua spoken in the Puno region of Peru. It was collected from January 2025 to March 2026 through a participatory design process with the National University of Altiplano of Puno, the Puno Regional Government and the community organisation Illariy Ch'aska. In total, 151 native speakers recorded through the localised Common Voice platform (<https://commonvoice.mozilla.org/qxp>), and community members validated the clips.

The corpus is released as two datasets on the Mozilla Data Collective under **CC0-1.0**:

| | Scripted Speech v26 (**SCS-26**) | Spontaneous Speech v4 (**SPS-4**) |
|---|---|---|
| Speech type | Read-aloud seed sentences | Free-form answers to open-ended questions |
| Includes earlier releases | v23, v24, v25 | v1, v2, v3 |
| Recorded duration | 34.88 h | 35.41 h |
| Validated duration | 31.18 h | 5.18 h (14.6 %) |
| Clips | 25,423 | 7,305 |
| Speakers | 81 | 110 |
| Prompts | 2,070 sentences | 150 questions |
| Avg. clip duration | 4.94 s | 17.45 s |
| Audio format / folder | `.mp3` in `clips/` | `.mp3` in `audios/` |
| Main metadata file | `validated.tsv` | `ss-corpus-qxp.tsv` |
| Repository (version v26 and v4) | [cmqio1wd400xynq079ifamac6](https://mozilladatacollective.com/datasets/cmqio1wd400xynq079ifamac6) | [cmqialqsu007bnr07e4se1rl8](https://mozilladatacollective.com/datasets/cmqialqsu007bnr07e4se1rl8) |
| Latest releases can be found at | [commonvoice.mozilla.org/datasets](https://commonvoice.mozilla.org/en/datasets) | [commonvoice.mozilla.org/datasets](https://commonvoice.mozilla.org/en/datasets) |

---

## 2. Release history on the Mozilla Data Collective

Common Voice publishes a new **cumulative** release every quarter. Each release contains every clip collected up to its data cut-off date, plus all validation votes cast by then. Every version has its own dataset page and download, and older versions remain available. Puno Quechua first appeared in Scripted Speech 23.0 and Spontaneous Speech 1.0, partway through the January 2025 – March 2026 collection campaign.

**Permanent entry point:** <https://commonvoice.mozilla.org/en/datasets>. This page lists every Common Voice dataset (Scripted and Spontaneous Speech, all languages and versions) and links to its download page on the Mozilla Data Collective. Links to individual dataset pages, such as those in the tables below, are specific to one version. If one stops working, search there for "Puno Quechua" or `qxp`.    

**Scripted Speech (`qxp`)**

| Release | Data cut-off | Announced | Clips | Speakers | Total h | Validated h | Dataset page |
|---|---|---|---:|---:|---:|---:|---|
| 23.0 | 2025-09-05 | 2025-09-17 | 9,033 | 14 | 11.63 | 9.88 | — |
| 24.0 | 2025-12-05 | 2025-12-16 | 9,033 | 14 | 11.63 | 9.88 | — |
| 25.0 | 2026-03-09 | March 2026 | 25,382 | 81 | 34.81 | 31.16 | [cmn2cxfs801dvo107r5rkzmvx](https://mozilladatacollective.com/datasets/cmn2cxfs801dvo107r5rkzmvx) |
| **26.0** | **2026-06-12** | **2026-06-19** | **25,423** | **81** | **34.88** | **31.18** | [**cmqio1wd400xynq079ifamac6**](https://mozilladatacollective.com/datasets/cmqio1wd400xynq079ifamac6) |
| 27.0 (latest) | 2026-09-11 | 2026-09-18 | 25,423 | 81 | 34.88 | 31.18 | [commonvoice.mozilla.org/datasets](https://commonvoice.mozilla.org/en/datasets) |

**Spontaneous Speech (`qxp`)**

| Release | Data cut-off | Announced | Clips | Speakers | Dataset page |
|---|---|---|---:|---:|---|
| 1.0 | 2025-09-05 | 2025-09-17 | 1,065 | 11 | — |
| 2.0 | 2025-12-05 | 2025-12-16 | 2,208 | 26 | — |
| 3.0 | 2026-03-09 | March 2026 | 7,286 | 110 | [cmn1pujk200uno107g6el5r9y](https://mozilladatacollective.com/datasets/cmn1pujk200uno107g6el5r9y) |
| **4.0** | **2026-06-12** | **2026-06-19** | **7,305** | **110** | [**cmqialqsu007bnr07e4se1rl8**](https://mozilladatacollective.com/datasets/cmqialqsu007bnr07e4se1rl8) |
| 5.0 (latest) | 2026-09-11 | 2026-09-18 | 7,305 | 110 | [commonvoice.mozilla.org/datasets](https://commonvoice.mozilla.org/en/datasets) |


> **Note:** For downloading Scripted Speech 26.0 and Spontaneous Speech 4.0, you can do it from the links above.

---

## 3. Evaluation rounds and their splits

We ran several evaluations while the corpus was still growing. Each round used the latest release available at the time and its own splits.

| Round | Dates (git) | Source releases | Split files |
|---|---|---|---|
| 0 — exploratory zero-shot | April 2026 | SCS-25 / SPS-3 | [`splits/sample_for_MMS_eval.tsv`](splits/sample_for_MMS_eval.tsv), [`splits/omni_transcripts_for_eval_small.tsv`](splits/omni_transcripts_for_eval_small.tsv) |
| 1 — first evaluation | April – May 2026 | SCS-25 / SPS-3 | [`splits/`](splits/) |
| **2 — second evaluation** | July 2026 | **SCS-26 / SPS-4** | [**`splits_cv26/`**](splits_cv26/) |

Rounds 0 and 1 are labelled SCS-25 / SPS-3 because they ran before 26.0 / 4.0 were published on 19 June 2026.

---

## 4. Out-of-domain evaluation set (`additional_data/`)

0.27 h of Puno Quechua speech from radio and social media, **used only for evaluation** (never for training or continued pre-training).

| Source (file prefix) | Clips |
|---|---:|
| `radio_novela_*` (radio drama, chapters I–II) | 81 |
| `entrevista_SC_*` (interview) | 24 |
| `nuqanchik_*` (broadcast programme) | 3 |
| `Rimasun_*` (social-media video) | 1 |
| **Total** | **109 clips, 0.27 h** |

- `wav/`: 16 kHz mono WAV segments.
- `additional_data_qxp.tsv`: `path`, `sentence` (manual reference transcription).

*TODO: original source and licence or permission for each recording, and who transcribed and checked the references.*

---

## 5. File formats

All files are tab-separated, UTF-8, with a header row. Read them with `quoting=3` (`csv.QUOTE_NONE`) because the transcripts contain apostrophes and quote characters.

**`splits_cv26/scripted/{train,dev,test}.tsv`**

| Column | Description |
|---|---|
| `client_id` | Pseudonymous Common Voice speaker ID |
| `path` | Clip filename in the SCS-26 `clips/` folder |
| `sentence_id` | Common Voice sentence ID |
| `sentence` | Original prompt text as released |
| `sentence_domain` | Domain tag of the prompt |
| `up_votes`, `down_votes` | Community validation votes |
| `age`, `gender` | Self-reported demographics |
| `duration_ms` | Clip duration |
| `transcript_for_asr` | **Reference used for training and evaluation** (curated) |

**`splits_cv26/spontaneous/{train,dev,test}.tsv`**

| Column | Description |
|---|---|
| `client_id` | Pseudonymous speaker ID (different from the scripted IDs) |
| `audio_id`, `audio_file` | Clip ID and filename in the SPS-4 `audios/` folder |
| `duration_ms` | Clip duration |
| `prompt_id`, `prompt` | Question the speaker answered |
| `transcription` | Validated human transcription (gold) or omniASR_LLM_7B output (silver) |
| `remove_disfluency` | Gold only: transcription with hesitation markers (e.g. `[ aaa]`) removed, where present |
| `transcript_for_asr` | **Reference used for training and evaluation**: lowercase, no punctuation |
| `gender` | Self-reported gender |
| `type` | `gold` or `silver` (see [Label tiers](#4-label-tiers)) |

---

## 6. Obtaining the audio

1. Create an account at the [Mozilla Data Collective](https://mozilladatacollective.com) and accept the data usage agreement. The agreement forbids attempting to re-identify contributors and redistributing the data. Access is granted immediately.
2. Download Scripted Speech and Spontaneous Speech for Puno Quechua (`qxp`). Use the links in [Section 2](#2-release-history-on-the-mozilla-data-collective), or find them from the permanent listing at <https://commonvoice.mozilla.org/en/datasets>.
3. Convert the audio to 16 kHz mono WAV and build the fairseq manifests as described in [`pipeline.sh`](../pipeline.sh).
4. Join the audio to the split files on `path` (scripted) or `audio_file` (spontaneous).

---

## 7. Licence and ethics

- Common Voice audio and metadata: **CC0-1.0**. Contributors agreed to the Common Voice terms and to an explicit consent statement in the open call (*"Entiendo que mis grabaciones se utilizarán para construir tecnología de voz de código abierto"*). Minors were excluded.
- Split files, curated transcripts and silver transcripts in this folder: CC0-1.0, the same as the source.
