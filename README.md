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
5. Writes `Images_All.csv`, `Summary_All.csv`, and `Skipped.csv` to the dataset root. `Summary_All.csv` restores the detailed analytics (group means/medians/modes, effect sizes, weights, XY table) from the earlier pipeline, explicitly excludes `RAW_00000` from statistical equations, and now starts with the contents of `Summary_Updated_Java.csv` (logistic attribute stats) separated by two blank rows from the remaining sections.
6. Column guides inside `Summary_All.csv` clarify what `value_a`–`value_d` represent for each section, including shrunken categorical weights that down-weight tiny/imbalanced samples using a reliability factor `n/(n+50)`.
7. Generates `Summary_Updated_Java.csv` and `Decision_Table_Java.csv` with the same logistic analysis that was previously in `SummaryGeneratorLogistic`, using only the in-memory records—no extra script needed.
8. Generates `Auto_Probabilities.csv`, a lightweight logistic-style probability table (based only on the extracted features) so no external `SummaryGeneratorLogistic` step is needed.

## Interpreting the outputs

### Summary_All.csv
- The file begins with the `LOGIT_SUMMARY` block (the contents of `Summary_Updated_Java.csv`). `value_a` = empirical flood rate, `value_b` = logit coefficient, `value_c` = odds ratio, `value_d` = samples; notes list the baseline, confidence bucket, standard error, and 95% CI.
- Two blank rows separate the `LOGIT_SUMMARY` from the downstream sections. The `NOTE` rows at the top restate what `value_a`–`value_d` mean in each section:
  - **STATS**: `value_a` (flood=true metric), `value_b` (flood=false metric), `value_c/value_d` noted per row. Cohen’s *d* is `(mean_true − mean_false) / pooled_std`, where `pooled_std = sqrt(((n1−1)*sd1^2 + (n2−1)*sd2^2) / (n1+n2−2))`. Post-mean rows exclude `RAW_00000` and use 3σ outlier removal.
  - **SEASONS / SHAPES**: `value_a` true count, `value_b` false count, `value_c` true rate, `value_d` unused.
  - **WEIGHTS**: numeric weights show Cohen’s *d* alongside overall mean/std/sample counts; categorical weights use a reliability shrink `n/(n+50)` so tiny or imbalanced groups cannot over-dominate.
  - **XY_TABLE / DECISION_RULE**: `XY_TABLE` includes its own column guide; `DECISION_RULE` note shows the scoring equation.

### Summary_Updated_Java.csv
- Mirrors the `LOGIT_SUMMARY` portion of `Summary_All.csv` in a standalone file.

### Decision_Table_Java.csv
- Lists only combinations that meet **all** thresholds: samples > 358, margin_of_error_95 ≤ 0.0954 (margin = 1.96 × standard_error, where `standard_error = sqrt(p*(1−p)/n)`), and empirical_flood_rate ≥ 0.5. Columns include the 95% CI bounds and the computed margin_of_error_95 so you can verify the filter directly.

### Auto_Probabilities.csv
- Uses a lightweight score: `score = z_raw_mean + 0.6*norm_black_diameter + 0.6*norm_white_diameter + season_bias + polarization_bias + shape_bias`; `probability = 1/(1+exp(−score))`. Here, `z_raw_mean` is z-scored against the dataset mean/std, diameters are normalized to their respective maxima, and the bias terms add small boosts/penalties based on season, polarization, and dominant shape.
