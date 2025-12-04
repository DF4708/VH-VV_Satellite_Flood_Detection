# Image_Java_Singular

This repository contains `Data_Extraction_M1_Optimized.java`, a Java program for processing SEN12FLOOD-style datasets.

## Usage

1. **Open a terminal in the repository root** (the same folder that contains `Data_Extraction_M1_Optimized.java`).
2. **Compile**:
   ```bash
   javac Data_Extraction_M1_Optimized.java
   ```
3. **Run** by passing the dataset root folder (defaults to the current directory if omitted):
   ```bash
   java Data_Extraction_M1_Optimized /path/to/dataset
   ```

If you prefer to compile from another directory, provide the full path to the source file, for example:
```bash
javac /workspace/Image_Java_Singular/Data_Extraction_M1_Optimized.java
```

### What happens automatically

When you run the program it performs the full pipeline without any extra scripts:

1. Cleans the dataset folder by removing non-TIFF artifacts (except Java/JSON files).
2. Parses `S1list.json` and `S2list.json` labels. If neither file is present, the tool still runs and assumes `flooding=false` for every image so CSVs are produced without manual post-processing. The parser warns if a JSON file is missing or yields zero matches to help pinpoint malformed entries.
3. Processes all `.tif` / `.tiff` images concurrently, scanning every subfolder under the dataset root (not just numeric folders).
4. Derives season directly from the `YYYYMMDD` portion embedded in each filename (e.g., `20190118` -> Winter) so seasonal counts and probabilities in `Summary_All.csv` stay aligned with the data.
5. Writes `Images_All.csv`, `Summary_All.csv`, and `Skipped.csv` to the dataset root. `Summary_All.csv` restores the detailed analytics (group means/medians/modes, effect sizes, weights, XY table) from the earlier pipeline, explicitly excludes `RAW_00000` from statistical equations, and now starts with the former `Summary_Updated_Java` (logistic attribute stats) separated by two blank rows from the remaining sections.
6. Column guides inside `Summary_All.csv` clarify what `value_a`–`value_d` represent for each section, and the header now exposes explicit columns for baseline, confidence, standard error, CI bounds, and 95% margins where applicable. Categorical weights down-weight tiny/imbalanced samples using a reliability factor `n/(n+50)` and the overall score is multiplied by a global confidence adjustment.
7. Generates `Decision_Table_Java.csv` with the same logistic analysis that was previously in `SummaryGeneratorLogistic`, using only the in-memory records—no extra script needed. The table now lists every season/polarization/shape combination (from extremely low to high confidence) and adds an explanatory note whenever the unstable flag is true. The standalone `Summary_Updated_Java.csv` is no longer written because its contents are embedded at the top of `Summary_All.csv`.
8. Generates `Auto_Probabilities.csv`, a confidence-weighted probability table that reuses the `Summary_All.csv` decision-rule score (with reliability scaling for categorical weights and a global confidence adjustment) and sorts rows from high to low probability. The header comment explains that the probability is the confidence-weighted chance an image is flooded given its attributes.

## Interpreting the outputs

### Summary_All.csv
- The file begins with the `LOGIT_SUMMARY` block (the former `Summary_Updated_Java`), with explicit columns for baseline, confidence_from_n, standard_error, CI bounds, and margin_of_error_95.
- Two blank rows separate the `LOGIT_SUMMARY` from the downstream sections. Remaining `NOTE` rows restate what `value_a`–`value_d` mean in each section:
  - **STATS**: `value_a` (flood=true metric), `value_b` (flood=false metric), `value_c/value_d` noted per row. Cohen’s *d* is `(mean_true − mean_false) / pooled_std`, where `pooled_std = sqrt(((n1−1)*sd1^2 + (n2−1)*sd2^2) / (n1+n2−2))`. Post-mean rows exclude `RAW_00000` and use 3σ outlier removal.
  - **SEASONS / SHAPES**: `value_a` total count, `value_b` flooding count, notes include the explicit true/false split.
  - **WEIGHTS**: numeric weights show Cohen’s *d* alongside overall mean/std/sample counts; categorical weights use a reliability shrink `n/(n+50)` so tiny or imbalanced groups cannot over-dominate and are weighted by representativeness.
  - **XY_TABLE / DECISION_RULE**: `XY_TABLE` includes its own column guide; `DECISION_RULE` note shows the scoring equation with reliability scaling.

### Decision_Table_Java.csv
- Lists every observed combination and includes a confidence label from *n* (Extremely Low → High). Unstable rows (tiny *n* or extreme rates) carry an explanatory note in the final column; sorting remains by flood probability and then by margin of error so the most reliable signals appear first. A `leans_toward` column flags whether the empirical rate favors predicting flood (`flood`) or no flood (`no_flood`).

### Auto_Probabilities.csv
- Uses the same decision-rule score described in `Summary_All.csv`: `score = confidence_adjusted(w_raw*z_raw_mean + w_bd*z_black_diameter + w_wd*z_white_diameter + w_season + w_pol + w_black_shape + w_white_shape); each categorical weight is multiplied by reliability n/(n+50); z_feature = (x - mean_all)/std_all; probability = 1/(1+exp(-score))`. The header comment reiterates that this probability is the confidence-weighted chance of flooding for the given attributes, and rows are sorted from highest to lowest probability.
