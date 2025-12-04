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
5. Writes `Images_All.csv`, `Summary_All.csv`, and `Skipped.csv` to the dataset root.
6. Generates `Summary_Updated_Java.csv` and `Decision_Table_Java.csv` with the same logistic analysis that was previously in `SummaryGeneratorLogistic`, using only the in-memory records—no extra script needed.
7. Generates `Auto_Probabilities.csv`, a lightweight logistic-style probability table (based only on the extracted features) so no external `SummaryGeneratorLogistic` step is needed.
