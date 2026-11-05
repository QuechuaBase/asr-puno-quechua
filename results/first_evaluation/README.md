# First evaluation (archived)

Results from the initial evaluation, kept for reference. They were superseded by the evaluation reported in the paper (see the main [README](../../README.md)) and **should not be cited as current results**.

Setup: Common Voice scripted/spontaneous splits from `data/splits/` and `data/splits_joh/`, manifests in `data/manifests/finetune/qxp_v2/` (test: 2,296 scripted and 360 spontaneous clips). Per-model hypothesis/reference files remain in `results/ft_*/` and `results/omnilingual/`; `summary.csv` here was produced by `eval/fairseq/analyze_results.py`.

| Model | Scripted WER | Scripted CER | Spontaneous WER | Spontaneous CER |
|-------|-------------:|-------------:|----------------:|----------------:|
| ft_cpt_validated | 1.19% | 0.19% | 13.61% | 1.73% |
| ft_cpt_silver | 2.11% | 0.30% | 3.15% | 0.41% |
| ft_xlsr_validated | 2.06% | 0.30% | 13.58% | 1.71% |
| ft_xlsr_silver | 4.36% | 0.57% | 6.68% | 0.81% |
| omniASR LLM 300M v2 | 7.07% | 0.89% | 11.98% | 1.54% |
| omniASR CTC 300M v2 | 24.47% | 3.05% | 20.22% | 2.43% |

**CPT** = continued pre-trained checkpoint; **XLSR** = public XLSR-128 300M baseline; **omniASR** = zero-shot Omnilingual baseline. **Validated** = gold-labeled data only; **Silver** = plus auto-transcribed spontaneous speech.
