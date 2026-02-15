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
