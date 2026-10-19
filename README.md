# Vision-language models: pulmonary nodule or vessel on chest CT

Code, model outputs, and images for the study

> Vision-Language Models Differentiating Pulmonary Nodules From Vessels on Chest CT: Answer-Option Order and Recognition of Missing or Noise-Only Images. [AUTHOR NAME]. [CITATION].

Ten vision-language models (seven open-weight, three commercial) were asked whether the structure at the centre of nine consecutive CT slices was a pulmonary nodule or a vessel. The question was asked with the two options in both orders, and again with noise images and with no image.

This repository contains the experiments reported in the paper. Nothing here comes from any hospital; all images are derived from the public LIDC-IDRI collection.

## Contents

| Path | What it is |
|---|---|
| `prompts/PROMPTS.md` | The two prompts, with their SHA-1 checksums |
| `images/ct/` | The nine 448 × 448 images of each structure given to the models (104 folders) |
| `manifest/` | Structure list, diameters, LIDC series identifiers, vessel candidates and their review |
| `results/run22_prompt_open.csv` | Replies of the open-weight models to CT images (final wording; Table 2) |
| `results/run22_prompt_frontier.csv` | Replies of the commercial models to CT images (final wording; Table 2) |
| `results/run23_retest_frontier.csv` | Second submission of 44 structures to the commercial models, both orders (`rep` = 2) |
| `results/run23_noise_open.csv`, `results/run23_noimg_open.csv` | Replies of the open-weight models to noise images and with no image (final wording; Table 3) |
| `results/run23_control_frontier.csv` | Replies of the commercial models to noise images and with no image, ten calls per order (final wording; Table 3) |
| `results/run23_noise_noimg_hand_verdicts.csv`, `results/run23_noise_statement_verdicts.csv` | Radiologist's classification of the 180 commercial replies and of the 31 replies containing the word "noise" (final wording) |
| `results/lidc_roi_results.csv`, `results/frontier_results.csv`, `results/control_stimuli_results.csv`, `results/control_noimg_fulltext.csv`, `results/frontier_control_results.csv` | The same experiments with the original wording ("low-dose"); Supplemental Methods and Tables S3–S5 |
| `results/frontier_raw/` | Archived request and response of every call to a commercial model |
| `results/run20_noise_hand_verdicts.csv`, `results/noise_statement_verdicts.csv`, `results/noimage_hand_verdicts.csv` | Radiologist's classifications for the original wording |
| `results/v011_tables.txt` | Output of `33_tables_v011.py` as used for the paper |
| `results/v011_effect_size.txt` | Output of `34_effect_size_v011.py`: the Youden index with patient-level bootstrap 95% CIs, and differences in accuracy between base and fine-tuned models |
| `results/v011_power.txt` | Output of `36_power_v011.py`: simulated power of the Fisher test for 39 nodules and 54 vessels (Methods, sample size) |
| `results/v011_acquisition.txt` | Output of `35_acquisition_v011.py`: scanner model, reconstruction kernel, and tube current of the 48 scans, and Table 1 by structure (needs the DICOM files) |
| `figs/` | Figure 1 (`fig1_ajr_flow`, selection of the structures), Figure 2 (`fig1_v011_nine_slices`), Figure 3 (`fig2_v011_forest`), and the scripts that draw them |
| `MODELS.md` | Model files, checksums, and software versions |

## Reproducing the tables

Python 3 with `scipy` is required.

```bash
python 33_tables_v011.py
```

This gives Table 2 (sensitivity, specificity, accuracy, the tests of discrimination and of option order), the analysis with one vessel per patient, the comparison of base and fine-tuned models, the repeatability of the commercial models, and Table 3. `30_analyze_run22.py` compares the two wordings of the question (Supplemental Methods, Results With the Original Wording); `28_reanalysis_v10.py` and `29_table3_and_audit.py` give the same tables for the original wording.

```bash
python 34_effect_size_v011.py
python figs/make_fig2_v011_forest.py
python figs/make_fig1_flow_ajr.py
```

`34_effect_size_v011.py` gives the Youden index for each model and order with 95% CIs from bootstrap resampling of patients, and the differences in accuracy between each fine-tuned model and its base model. `figs/make_fig2_v011_forest.py` draws Figure 3 after checking its counts against Table 2 and its Youden index values against the counts. `figs/make_fig1_v011.py` draws Figure 2 from the images in `images/ct/`. `figs/make_fig1_flow_ajr.py` draws Figure 1 from the files in `manifest/` and checks each count; it needs Graphviz (`dot`).

## Two wordings of the question

The experiments were first run with a question that described the images as coming from "a low-dose chest CT". Because 21 of the 48 scans were not low-dose acquisitions, every experiment was repeated with the word "low-dose" removed, and the paper reports the repeated experiments. Both sets of results are here; `prompts/PROMPTS.md` gives both wordings with their checksums.

## Structures: 104 run, 93 analysed

The models were first run on 104 structures: 50 nodules and 54 vessels. The vessels measured 4.5–14.3 mm. Eleven nodules were larger than 14.3 mm and were excluded so that nodules and vessels covered the same diameter range. The paper reports the remaining 93 structures (39 nodules, 54 vessels). The original-wording result files keep all 104; the final-wording files contain the 93. The analysis scripts apply the restriction. Of the 50 structures submitted a second time to the commercial models, 44 are among the 93.

## Re-running the experiment

1. Download the LIDC-IDRI series listed in `manifest/series_to_download.txt` from The Cancer Imaging Archive (`03_download_series.py`).
2. Render the images: `05_render_nodules.py` for nodules, `06_find_foils.py`, `08_select_foils.py`, and `09_render_foils.py` for vessels. The images in `images/ct/` are the output of this step.
3. Generate the noise images with `20_build_control_stimuli.py` (random seeds 1–20). Their checksums are in `manifest/patches_control_sha256.txt`.
4. Obtain the model files listed in `MODELS.md` and check their SHA-256.
5. Run `run_roi_batch.py` (llama.cpp models), `run_roi_batch_hf.py` (Lingshu-7B), `run_frontier.py` (commercial models, API keys read from environment variables), `21_run_control_logprob.py` and `27_run_noimg_fulltext.py` (noise and no-image inputs for the open-weight models).

Replies of the commercial models are not deterministic and the models may be changed or withdrawn by their vendors, so a re-run will not reproduce those files exactly.

## Notes on the files

- Local paths in the `command` column of the result files have been replaced with `<MODELS_DIR>` and `<HOME>`.
- API keys and authorization headers have been removed from `results/frontier_raw/` and replaced with `<REDACTED>`. Nine archives were affected. In some of them the match was a chance substring inside an encrypted reasoning string rather than a key; that substring was replaced as well. Model outputs are unchanged.
- In the noise and no-image runs, the two Qwen3-VL models continued to generate text after the end of their reply ("Hello … Hi there … How are you?"). The answer and diameter precede this text and are not affected.
- The run scripts also contain code paths for conditions that are not part of this paper.
- Original DICOM files are not included.

## Licence and attribution

Code: MIT (`LICENSE`). Images and model outputs: CC BY 4.0 (`LICENSE-DATA.md`).

The images are derived from LIDC-IDRI (CC BY 3.0): Armato SG III, McLennan G, Bidaut L, et al. Data From LIDC-IDRI. The Cancer Imaging Archive, 2015. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX

The authors acknowledge the National Cancer Institute and the Foundation for the National Institutes of Health, and their critical role in the creation of the free publicly available LIDC/IDRI Database used in this study.
