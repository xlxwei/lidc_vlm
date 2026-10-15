# Prompts

Each model received nine images followed by one of two prompts, which differ only in the order of the two options. The paper reports the final wording (files `run22_*` and `run23_*`). The original wording, which read "from a low-dose chest CT", was used for the files `lidc_roi_results.csv`, `frontier_results.csv`, `control_stimuli_results.csv`, `control_noimg_fulltext.csv`, and `frontier_control_results.csv`; its results are in the supplement.

## Final wording

### Nodule-first (SHA-1, first 12 characters: 958c37a4c6de)

> These are 9 consecutive axial slices (superior to inferior, 1.25 mm spacing) from a chest CT, cropped to a 64 mm field of view centred on one structure in the lung. Is the structure at the centre of the images a pulmonary nodule or a cross-section of a blood vessel? Answer with one word, Nodule or Vessel, then one short sentence of reasoning, then the estimated diameter in mm.

### Vessel-first (666ec3e5d79b)

> These are 9 consecutive axial slices (superior to inferior, 1.25 mm spacing) from a chest CT, cropped to a 64 mm field of view centred on one structure in the lung. Is the structure at the centre of the images a cross-section of a blood vessel or a pulmonary nodule? Answer with one word, Vessel or Nodule, then one short sentence of reasoning, then the estimated diameter in mm.

## Original wording

### Nodule-first (b046abfad45a)

> These are 9 consecutive axial slices (superior to inferior, 1.25 mm spacing) from a low-dose chest CT, cropped to a 64 mm field of view centred on one structure in the lung. Is the structure at the centre of the images a pulmonary nodule or a cross-section of a blood vessel? Answer with one word, Nodule or Vessel, then one short sentence of reasoning, then the estimated diameter in mm.

### Vessel-first (b3a242d23748)

> These are 9 consecutive axial slices (superior to inferior, 1.25 mm spacing) from a low-dose chest CT, cropped to a 64 mm field of view centred on one structure in the lung. Is the structure at the centre of the images a cross-section of a blood vessel or a pulmonary nodule? Answer with one word, Vessel or Nodule, then one short sentence of reasoning, then the estimated diameter in mm.

The `prompt_sha1` column of each results file records which prompt a call used.
