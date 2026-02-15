# Image_Java_Singular

**ALL RIGHTS RESERVED**

## Summary

The methodology centers on automated, data-driven image analysis. The Java script iterates through numbered folders, pairing each TIFF image with its corresponding GeoJSON file by extracting matching date codes. Only images whose GeoJSON contains a Flooding flag (`true` or `false`) are processed. Each qualifying image is read directly from its raster—bypassing color-model conversions—to compute a full greyscale histogram and identify contiguous black and white regions. The algorithm then measures each region’s diameter, width, height, and approximates its geometric shape via convex-hull and roundness analysis. Results are consolidated into structured CSV outputs (summary, per-color counts, and skip logs) suitable for further statistical or spatial analysis.

The script’s performance evaluation shows that it scales linearly with image size and efficiently handles large datasets. The main raster-processing method, `buildHistogramAndMasksFromRaster`, operates in **O(W × H)** time and space, where **W** and **H** are the image width and height, since every pixel is read once to compute grayscale and masks. The connected-component search (`largestComponent`) runs in **O(P)**, where **P** is the number of pixels belonging to the black or white region being explored. Geometry calculations are dominated by the convex hull (**O(M log M)**) and rotating calipers diameter computation (**O(M)**), where **M** is the number of points in the region’s boundary. GeoJSON parsing (`parseGeojsonFloodFlags`) runs in **O(S)**, where **S** is the number of characters in the file, and CSV export is **O(F × U)**, with **F** images processed and **U** unique grayscale values. Overall runtime grows linearly with total pixels processed, making the algorithm computationally efficient and memory-stable for large-scale folder analysis.

**Source Images:** https://ieee-dataport.org/open-access/sen12-flood-sar-and-multispectral-dataset-flood-detection#

---

## Running the program

- Compile & run in the ROOT folder (the one that contains image folders `1/…/134`) by using `cd` in your terminal (shell, PowerShell, or CMD), then compile with `javac Filename.java` and run with `java Filename`.

---

## Simplified inputs & expected outputs

### Inputs

- A directory folder with numeric subfolders `1/…/134/`.
- In each folder: TIFF images (`.tif/.tiff`) and GeoJSON(s) numbered by geographic identifier.

### Outputs (in directory)

#### 1) `Images_All.csv` — per-image feature table

Each row corresponds to one processed image.

**Core columns**
- `image_name`
- `folder_name` (numeric folder or `ROOT`)
- `polarization` (`VV` | `VH` | `OTHER`)
- `flooding` (`true` | `false`)
- `season` (`Winter` | `Spring` | `Summer` | `Fall` | `Unknown`)
- `raw_mean` (mean of non-zero pixels)

**Black component feature columns** (selected connected component)
- `black_size`
- `black_width`
- `black_height`
- `black_diameter` (bbox diagonal)
- `black_shape`

**White component feature columns**
- `white_size`
- `white_width`
- `white_height`
- `white_diameter`
- `white_shape`

**Derived column**
- `dominant_shape`

**Histogram columns**
- `RAW_00000`, `RAW_00001`, ... up to the maximum observed raw key  
- Only keys that occur anywhere in the dataset are emitted as columns.  
- Missing values for a given image are written as `0`.

#### 2) `Summary_All.csv` — multi-section dataset report

This file is a single CSV that contains multiple labeled sections. (It’s designed to be human-readable *and* parseable with light scripting.)

Sections include:

##### `STATS`
- Aggregate statistics over `raw_mean`
- Includes both:
  - unfiltered stats, and
  - outlier-filtered stats (based on z-score filtering; i.e., restricting to values inside a standard deviation band)
- Includes a “post-mode” concept based on aggregate RAW histogram frequency (a distribution-wide mode-like measure)

##### `SEASONS`
- Counts and flooding rates by `season` (and sometimes by `polarization` depending on your generator’s exact grouping)

##### `SHAPES`
- Counts and flooding rates by shape combinations (black/white and/or dominant), as implemented by the summarizer

##### `WEIGHTS`
- Coefficients used in the scoring rule:
  - numeric feature weights (e.g., raw_mean, diameters)
  - categorical offsets (e.g., season, polarization, shape terms)

##### `XY_TABLE`
- Empirical flood probabilities by:
  - `season`, `polarization`, `black_shape`, `white_shape`
- Often computed only for observations inside a bounded z-score region to reduce distortion from extreme outliers.

##### `DECISION_RULE`
- Human-readable spec of the model form:
  - standardized feature construction
  - linear score
  - logistic transform

#### 3) `Skipped.csv` — trace log

Each row indicates an image that was not processed:
- `image_name`
- `folder_name`
- `reason`

---

## Simplified high-level architecture

Single self-contained Java program organized into three subsystems:

1. **Discovery & gating**
   1. Traverse folders `1..134`.
   2. Parse all GeoJSONs in a folder → build a `Map<YYYYMMDD, boolean>` where the value is `true` if **ANY** feature is flagged true.
   3. For each TIFF: extract last date in filename → require a flood entry for that date.
   4. If no date, or no flood entry, the image is skipped.

2. **Image analysis pipeline**
   1. Raster reader (no `getRGB()`, no color-model conversion): stream rows directly from the `Raster` supporting `BYTE/USHORT/INT/FLOAT/DOUBLE`.
   2. For each pixel:
      1. Normalize channels to `[0..1]`.
      2. Compute luma **Y** (BT.601): `Y = 0.299R + 0.587G + 0.114B`.
      3. Treat interior `alpha==0` as white; border transparency ignored to avoid halo artifacts.
      4. Quantize to `0..255` → update 256-bin histogram; build black (`Y<=19`) and white (`Y>=236` or interior transparent) boolean masks.
   3. Connected components (8-connectivity) to find the largest black and white components.
   4. Geometry:
      1. Bounding box width/height.
      2. Convex hull (monotone chain); rotating calipers for Euclidean diameter.
      3. Perimeter/area → roundness; aspect ratio.
      4. Heuristic shape classification from hull vertices + roundness + aspect + fill ratio.

3. **Export**
   1. Accumulate all unique hex greys across images.
   2. Write/overwrite CSVs (headers first, then final tables).
   3. Log reasons for skips and a histogram at the end (during earlier versions); final v6 writes rows and prints progress.

---

## Some design decisions

### 1) Transparent interior = white
To avoid border artifacts, an `alpha==0` pixel only counts as white when it’s not within a frame of border pixels from the image edges. (We used `border=2px`.)

### 2) Black/white thresholds
- Black = `Y <= 19`
- White = `Y >= 236` or interior transparent

To avoid areas too small and to account for clouds that can fluctuate in radiance, the “top 20 blackest / whitest” pixels on an 8-bit scale were accepted, provided they remained stable under quantization.

### 3) Largest connected component & geometry
- Connectivity: 8-way (captures diagonals, more natural for raster blobs).
- Metrics:
  - Euclidean diameter from rotating calipers on the convex hull (tight upper bound).
  - Width, height from the component’s axis-aligned bounding box.
  - Roundness = `4πA/P²`, aspect = `max(width,height)/min(width,height)`, vertex count of hull.

### 4) Shape classification (coarse)
Heuristic rules (simple, fast, stable under raster jitter):

- Triangle if a hull has 3 vertices.
- Circle if roundness high (≥ 0.78) and aspect ≈ 1.
- Ellipse/Oval if roundness moderately high with aspect > 1.
- Quadrilaterals → check right angles, equal sides, and parallelism patterns:
  - Square / Rectangle / Rhombus / Parallelogram / Trapezium.
- Crescent if fill ratio of blob inside its AABB is low yet hull is “many-sided”.
- Else default to Rectangle.

---

## Simplified data flow (per image)

```
TIFF -> ImageIO.read()
    -> Raster (width, height, bands, transferType)
    -> For each row:
           read pixels (int[] / float[] / double[])
           normalize channels -> [0..1]
           compute Y (luma)
           handle transparency rule
           quantize -> histogram[0..255]
           update masks[black/white]
    -> largestComponent(black), largestComponent(white)
         -> computeGeometry(...) for both
         -> classifyShape(...)
         -> write summary row
         -> add per-hex counts into image record

At the end:
- Build the complete set of unique hex greys.
- Output All_Images_Counts.csv with columns aligned for all images.
- Output Summary.csv, Skipped.csv.
```

---

## Error handling & “discovered vs processed” gap

Common skip reasons and how we addressed them:

- No date in filename → skip (no way to correlate).
- No matching GeoJSON (date key missing) → skip (no way to contribute to predictability).
- `ImageIO` returned `null` → can occur with unsupported TIFF; adding the TwelveMonkeys plugin could fix, but violates constraints.
- `OutOfMemoryError` on extremely large rasters → handled by catching and logging; the algorithm itself is scanline-based to minimize memory.

---

## Performance & complexity

### Computation
- Single pass over the raster: **O(W×H)** per image.
- Connected component BFS: **O(#pixels in mask)**. Performed twice (black/white) with linear performance **O(N)**.
- Convex hull: **O(N log N)** with `N ≪ W×H` (`N` = component pixels); in practice hull points are far fewer than component pixels.
- Rotating calipers: **O(H)** where `H` is hull size.

---

## Example performance execution

- Using **Apple Silicon M1 Max** with **10 core CPU (2E/8P)**, **32 core GPU**, **16 core NPU/ANE**, **64GB RAM**, and **1TB SSD**.

```
1  CMD% time java Data_Extraction_M1_Optimized.java
2  [INFO] Cleaning directory tree...
3  [INFO] Cleaning complete.
4  [INFO] Parsing JSON labels from: /Users/...
5  [INFO] Parsed 285 labeled entries from S1list.json
6  [INFO] Parsing JSON labels from: /Users/...
7  [INFO] Parsed 90 labeled entries from S2list.json
8  [INFO] Flood entries loaded from JSON: 375
9  [INFO] TIF/TIFF images discovered (potential): 9208
10 [INFO] Using thread pool with 8 worker threads.
11 [PROGRESS] Processed 9208 / 9208 images (100.0%)
12 [INFO] Image processing complete.
13 [INFO] Records: 8325, Skipped: 883
14 [INFO] Images_All.csv, Summary_All.csv, and Skipped.csv written in: /Users/...
15 [INFO] Auto_Probabilities.csv written with lightweight logistic-style scores in: /Users/...
16 java Data_Extraction_M1_Optimized.java  502.88s user 8.75s system 681% cpu 1:15.07 total
   a. 1 minute, 15 seconds, and 7 nanoseconds. Roughly ~85.13% of allotted 8 P-core's utilized.
```

---

## Implementation notes (key methods)

- `extractNormalizedDateKey(String)` — tolerant date extraction.
- `parseGeojsonFloodFlags(Path)` — scans entire GeoJSON text; `interpretFloodValue(String)` coalesces true/false encodings.
- `buildHistogramAndMasksFromRaster(BufferedImage, int border)` — core raster path; handles `BYTE/USHORT/INT/FLOAT/DOUBLE` uniformly.
- `largestComponent(boolean[][])` — BFS, 8-connectivity.
- `computeGeometry(List<Point>)` — bbox, hull, roundness, aspect, rotating calipers.
- `classifyShape(...)` and `classifyQuadrilateral(...)` — rule-based approximations.
- CSV helpers write UTF-8 with escaping.
