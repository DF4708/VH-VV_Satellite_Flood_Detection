import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.DataBuffer;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;

import java.nio.charset.StandardCharsets;
import java.nio.file.*;

import java.util.*;
import java.util.concurrent.*;

/**
 * Data_Extraction_M1_Optimized
 *
 * WHAT:
 *   Multi-core Java-only extractor for SEN12FLOOD-style data using S1list.json and S2list.json.
 *
 * HOW:
 *   - Reads S1list.json and S2list.json and builds:
 *       Sentinel filename (without extension) -> FLOODING true/false.
 *   - Scans:
 *       * All numeric subfolders under the root path (e.g., 0001, 33, 120).
 *       * The root folder itself (for test TIFFs).
 *   - Builds a job list of all .tif/.tiff images, then processes them in parallel
 *     with a fixed thread pool (tuned for Apple M1 Max: 8 threads).
 *   - For each image:
 *       * Finds the flood flag by matching base name against JSON filenames.
 *       * Derives polarization ("VV", "VH", or "OTHER").
 *       * Derives season from the YYYYMMDD date in the filename.
 *       * Reads pixels via Raster (supports 8/16-bit TIFFs with default ImageIO).
 *       * Computes:
 *           - RAW grayscale histograms: RAW_00000–RAW_65535 (no normalization).
 *           - Largest contiguous black/white components using two-pass thresholds,
 *             with a final fallback so every image gets a black and white shape.
 *             White and black shapes do not overlap (black excludes white region).
 *           - Component width / height / diameter and heuristic shape labels.
 *           - Per-image mean RAW value (based only on non-zero pixels).
 *
 *   - Writes three CSVs in the root folder:
 *       1) Images_All.csv
 *          One row per image (VV and VH), with:
 *            - image_name, folder_name, polarization, flooding, season
 *            - raw_mean (over non-zero pixels)
 *            - black_* and white_* geometry / shape
 *            - dominant_shape
 *            - RAW_XXXXX counts for all RAW values that appear in the dataset.
 *
 *       2) Summary_All.csv
 *          High-level data quality and distribution summary:
 *            - COUNTS  : total image counts by polarization, by flooding.
 *            - SEASONS : season counts and true_rates
 *            - SHAPES  : shape counts by flooding
 *            - WEIGHTS : weights for raw_mean, black_diameter, white_diameter,
 *                        season dummies, black_shape_*, white_shape_*, and polarization.
 *            - XY_TABLE: empirical probabilities by season, polarization, black_shape, white_shape
 *                        (within |z_raw_mean| <= 1).
 *            - DECISION_RULE : a row describing the score -> probability formula.
 *
 *       3) Skipped.csv
 *          One row per image that was not processed (no label, TIFF decoding error, etc).
 *
 *   - After CSV generation, automatically writes Auto_Probabilities.csv with lightweight
 *     logistic-style scores using only standard JDK classes (no external dependencies).
 */
public class Data_Extraction_M1_Optimized {

    // ---------- Helper types ----------

    private static class Component {
        int minX, maxX, minY, maxY;
        int pixelCount;

        Component() {
            minX = Integer.MAX_VALUE;
            maxX = Integer.MIN_VALUE;
            minY = Integer.MAX_VALUE;
            maxY = Integer.MIN_VALUE;
            pixelCount = 0;
        }

        void update(int x, int y) {
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
            pixelCount++;
        }

        int getWidth() {
            return (pixelCount == 0) ? 0 : (maxX - minX + 1);
        }

        int getHeight() {
            return (pixelCount == 0) ? 0 : (maxY - minY + 1);
        }

        double getDiameter() {
            if (pixelCount == 0) return 0.0;
            int dx = maxX - minX;
            int dy = maxY - minY;
            return Math.sqrt((double) dx * dx + (double) dy * dy);
        }
    }

    private static class ImageAnalysis {
        double rawMean;
        double rawMeanNoZero;
        String blackShape;
        int blackWidth;
        int blackHeight;
        double blackDiameter;
        String whiteShape;
        int whiteWidth;
        int whiteHeight;
        double whiteDiameter;
        Map<String, Integer> rawCounts;
        String dominantShape;
    }

    private static class MasksAndComponents {
        Component blackComponent;
        Component whiteComponent;
        boolean[][] blackMask;
        boolean[][] whiteMask;

        MasksAndComponents(Component blackComponent, Component whiteComponent,
                           boolean[][] blackMask, boolean[][] whiteMask) {
            this.blackComponent = blackComponent;
            this.whiteComponent = whiteComponent;
            this.blackMask = blackMask;
            this.whiteMask = whiteMask;
        }
    }

    private static class ImageJob {
        final Path imagePath;
        final String folderName;
        ImageJob(Path imagePath, String folderName) {
            this.imagePath = imagePath;
            this.folderName = folderName;
        }
    }

    private static class ImageRecord {
        String imageName;
        String folderName;
        boolean flooding;
        String season;
        String polarization;
        double rawMean;
        double rawMeanNoZero;
        String blackShape;
        int blackWidth;
        int blackHeight;
        double blackDiameter;
        String whiteShape;
        int whiteWidth;
        int whiteHeight;
        double whiteDiameter;
        String dominantShape;
        Map<String, Integer> rawCounts;
    }

    private static class SkipRecord {
        String imageName;
        String folderName;
        String reason;

        SkipRecord(String imageName, String folderName, String reason) {
            this.imageName = imageName;
            this.folderName = folderName;
            this.reason = reason;
        }
    }

    private static class JobResult {
        ImageRecord record;
        SkipRecord skip;

        JobResult(ImageRecord record, SkipRecord skip) {
            this.record = record;
            this.skip = skip;
        }
    }

    private static class CategoryStats {
        String attribute;
        String category;
        String baselineCategory;
        boolean isBaseline;
        int samples;
        double empiricalFloodRate;
        double standardError;
        double ciLow95;
        double ciHigh95;
        double logitCoefficient;
        double oddsRatio;
        double marginOfError95;
        String confidenceFromN;
    }

    private static class DecisionCombo {
        String season;
        String polarization;
        String blackShape;
        String whiteShape;
        int samples;
        int floodCount;
        double empiricalFloodRate;
        double standardError;
        double marginOfError95;
        double ciLow95;
        double ciHigh95;
        String confidenceFromN;
        boolean unstableExtremeFlag;
    }

    private static class LogisticModel {
        Map<String, Integer> featureIndex;
        int rawMeanIndex;
        int blackDiameterIndex;
        int numFeatures;
        double[] weights;
    }

    private static class LogisticBundle {
        List<CategoryStats> stats;
        List<DecisionCombo> combos;
    }

    // ---------- Main entry point ----------

    public static void main(String[] args) {
        Path rootFolder = (args.length > 0)
                ? Paths.get(args[0]).toAbsolutePath()
                : Paths.get("").toAbsolutePath();

        System.out.println("[INFO] Root folder: " + rootFolder);

        // Clean directory tree (integrated former Data_Cleaner)
        System.out.println("[INFO] Cleaning directory tree...");
        cleanDirectory(rootFolder.toFile());
        System.out.println("[INFO] Cleaning complete.");

        Path s1JsonPath = rootFolder.resolve("S1list.json");
        Path s2JsonPath = rootFolder.resolve("S2list.json");

        Map<String, Boolean> floodBySentinelName = new HashMap<>();

        try {
            if (Files.exists(s1JsonPath)) {
                parseFloodJsonFile(s1JsonPath, floodBySentinelName);
            } else {
                System.out.println("[WARN] Missing S1list.json at: " + s1JsonPath);
            }
            if (Files.exists(s2JsonPath)) {
                parseFloodJsonFile(s2JsonPath, floodBySentinelName);
            } else {
                System.out.println("[WARN] Missing S2list.json at: " + s2JsonPath);
            }
        } catch (IOException e) {
            System.err.println("[FATAL] Failed to read S1list.json/S2list.json: " + e.getMessage());
            return;
        }

        boolean noLabelsFound = floodBySentinelName.isEmpty();
        if (noLabelsFound) {
            System.err.println("[WARN] No FLOODING labels found in S1list.json or S2list.json. Proceeding with flooding=false for all images.");
        }
        final boolean assumeFloodFalse = noLabelsFound;

        System.out.println("[INFO] Flood entries loaded from JSON: " + floodBySentinelName.size());

        // Gather all image jobs anywhere under the root folder.
        List<ImageJob> jobs = new ArrayList<>();
        int discovered = gatherJobs(rootFolder, jobs);
        System.out.println("[INFO] TIF/TIFF images discovered (potential): " + discovered);

        if (jobs.isEmpty()) {
            System.out.println("[WARN] No TIF/TIFF images found. Exiting.");
            return;
        }

        // Multi-core processing: use 8 threads on M1 Max (8 performance cores).
        final int THREAD_COUNT = 8;
        ExecutorService pool = Executors.newFixedThreadPool(THREAD_COUNT);
        System.out.println("[INFO] Using thread pool with " + THREAD_COUNT + " worker threads.");

        List<Future<JobResult>> futures = new ArrayList<>();
        for (ImageJob job : jobs) {
            futures.add(pool.submit(new Callable<JobResult>() {
                @Override
                public JobResult call() {
                    return processImageJob(job, floodBySentinelName, assumeFloodFalse);
                }
            }));
        }

        pool.shutdown();

        List<ImageRecord> records = new ArrayList<>();
        List<SkipRecord> skips = new ArrayList<>();

        int totalJobs = futures.size();
        int processedJobs = 0;

        for (Future<JobResult> future : futures) {
            try {
                JobResult result = future.get();
                if (result != null) {
                    if (result.record != null) records.add(result.record);
                    if (result.skip != null)   skips.add(result.skip);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                System.err.println("[FATAL] Interrupted while waiting for jobs: " + e.getMessage());
                return;
            } catch (ExecutionException e) {
                System.err.println("[WARN] Uncaught error in worker: " + e.getCause());
            }

            processedJobs++;
            if (processedJobs % 10 == 0 || processedJobs == totalJobs) {
                double pct = (100.0 * processedJobs) / totalJobs;
                System.out.printf("\r[PROGRESS] Processed %d / %d images (%.1f%%)", processedJobs, totalJobs, pct);
            }
        }
        System.out.println();
        System.out.println("[INFO] Image processing complete.");
        System.out.println("[INFO] Records: " + records.size() + ", Skipped: " + skips.size());

        if (records.isEmpty()) {
            System.out.println("[WARN] No images produced data rows. Check labels and TIFF formats.");
        }

        // Build global RAW code set.
        Set<String> allRawCodes = new TreeSet<>();
        for (ImageRecord rec : records) {
            allRawCodes.addAll(rec.rawCounts.keySet());
        }

        // Build main per-image CSV rows.
        List<List<String>> imagesAllRows = buildImagesAllRows(records, allRawCodes);

        LogisticBundle logisticBundle = computeLogisticBundle(records);

        // Build summary CSV sections.
        WeightContext weightContext = computeWeights(records);

        List<List<String>> summaryAllRows = buildSummaryAllRows(records, logisticBundle, weightContext);

        // Build skipped CSV rows.
        List<List<String>> skippedRows = buildSkippedRows(skips);

        // Write CSVs in root folder.
        Path imagesAllPath = rootFolder.resolve("Images_All.csv");
        Path summaryAllPath = rootFolder.resolve("Summary_All.csv");
        Path skippedPath   = rootFolder.resolve("Skipped.csv");

        try {
            writeCsv(imagesAllPath, imagesAllRows);
            writeCsv(summaryAllPath, summaryAllRows);
            writeCsv(skippedPath, skippedRows);
        } catch (IOException e) {
            System.err.println("[FATAL] Failed to write CSV files: " + e.getMessage());
            return;
        }

        System.out.println("[INFO] Images_All.csv, Summary_All.csv, and Skipped.csv written in: " + rootFolder);

        try {
            writeLogisticSummaries(rootFolder, records, logisticBundle);
        } catch (IOException e) {
            System.err.println("[WARN] Failed to write logistic summaries: " + e.getMessage());
        }

        try {
            writeAutoProbabilities(rootFolder, records, weightContext);
        } catch (IOException e) {
            System.err.println("[WARN] Failed to write Auto_Probabilities.csv: " + e.getMessage());
        }
    }

    // ----------
    // ---------- Cleaning (formerly Data_Cleaner) ----------
    // ----------

    private static void cleanDirectory(File folder) {
        File[] files = folder.listFiles();
        if (files == null) return;

        for (File file : files) {
            if (file.isDirectory()) {
                cleanDirectory(file);
                continue;
            }

            String name = file.getName();
            boolean valid =
                    name.endsWith("VV.tif") || name.endsWith("VH.tif") ||
                            name.endsWith(".java")  || name.endsWith(".json");

            if (!valid) {
                // best-effort delete; ignore failure
                file.delete();
            }
        }
    }

    // ---------- Job gathering ----------

    private static int gatherJobs(Path rootFolder, List<ImageJob> jobs) {
        int count = 0;

        try {
            // Walk the entire tree so nested folders (not only numeric) are picked up.
            try (java.util.stream.Stream<Path> stream = Files.walk(rootFolder)) {
                for (Iterator<Path> it = stream.iterator(); it.hasNext(); ) {
                    Path entry = it.next();
                    if (Files.isRegularFile(entry)) {
                        String nameLower = entry.getFileName().toString().toLowerCase(Locale.ROOT);
                        if (nameLower.endsWith(".tif") || nameLower.endsWith(".tiff")) {
                            Path parent = entry.getParent();
                            String folderName = (parent != null && !parent.equals(rootFolder))
                                    ? rootFolder.relativize(parent).toString()
                                    : "";
                            jobs.add(new ImageJob(entry, folderName));
                            count++;
                        }
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("[WARN] Failed to scan folders for TIF/TIFF files: " + e.getMessage());
        }

        return count;
    }

    // ---------- JSON parsing ----------

    private static void parseFloodJsonFile(Path jsonPath, Map<String, Boolean> floodBySentinelName) throws IOException {
        System.out.println("[INFO] Parsing JSON labels from: " + jsonPath);

        String json = Files.readString(jsonPath, StandardCharsets.UTF_8);

        // Match FLOODING/filename pairs within the same JSON object (stop at the first closing brace).
        int before = floodBySentinelName.size();

        java.util.regex.Pattern floodingThenFile = java.util.regex.Pattern.compile(
                "\\\"FLOODING\\\"\\s*:\\s*(true|false)[^}]*?\\\"filename\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"",
                java.util.regex.Pattern.CASE_INSENSITIVE | java.util.regex.Pattern.DOTALL);

        java.util.regex.Pattern fileThenFlooding = java.util.regex.Pattern.compile(
                "\\\"filename\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"[^}]*?\\\"FLOODING\\\"\\s*:\\s*(true|false)",
                java.util.regex.Pattern.CASE_INSENSITIVE | java.util.regex.Pattern.DOTALL);

        addMatches(json, floodingThenFile, floodBySentinelName);
        addMatches(json, fileThenFlooding, floodBySentinelName);

        int added = floodBySentinelName.size() - before;
        if (added == 0) {
            System.out.println("[WARN] No FLOODING entries found in " + jsonPath.getFileName() + " (pattern mismatch).");
        } else {
            System.out.println("[INFO] Parsed " + added + " labeled entries from " + jsonPath.getFileName());
        }
    }

    private static void addMatches(String json, java.util.regex.Pattern pattern, Map<String, Boolean> floodBySentinelName) {
        java.util.regex.Matcher matcher = pattern.matcher(json);
        while (matcher.find()) {
            // Group order flips depending on pattern, so test group count.
            boolean firstIsFlood = matcher.group(1).equalsIgnoreCase("true") || matcher.group(1).equalsIgnoreCase("false");
            boolean flooding;
            String filename;
            if (firstIsFlood) {
                flooding = Boolean.parseBoolean(matcher.group(1));
                filename = matcher.group(2);
            } else {
                filename = matcher.group(1);
                flooding = Boolean.parseBoolean(matcher.group(2));
            }

            String baseName = stripExtension(filename.trim());
            floodBySentinelName.put(baseName, flooding);
        }
    }

    private static String stripExtension(String filename) {
        int dot = filename.lastIndexOf('.');
        if (dot < 0) return filename;
        return filename.substring(0, dot);
    }

    // ---------- Image job processing ----------

    private static JobResult processImageJob(ImageJob job, Map<String, Boolean> floodBySentinelName, boolean assumeFloodFalse) {
        Path imagePath = job.imagePath;
        String folderName = job.folderName;
        String fileName = imagePath.getFileName().toString();
        String baseName = stripExtension(fileName);

        Boolean floodFlag = null;
        if (floodBySentinelName.isEmpty() && assumeFloodFalse) {
            floodFlag = Boolean.FALSE;
        } else {
            for (Map.Entry<String, Boolean> e : floodBySentinelName.entrySet()) {
                String key = e.getKey();
                if (baseName.contains(key)) {
                    floodFlag = e.getValue();
                    break;
                }
            }

            if (floodFlag == null) {
                return new JobResult(null, new SkipRecord(fileName, folderName, "No matching FLOODING label in JSON."));
            }
        }

        BufferedImage image;
        try {
            image = javax.imageio.ImageIO.read(imagePath.toFile());
        } catch (IOException e) {
            return new JobResult(
                    null,
                    new SkipRecord(fileName, folderName, "Failed to read image: " + e.getMessage())
            );
        }

        if (image == null) {
            return new JobResult(
                    null,
                    new SkipRecord(fileName, folderName,
                            "Unsupported image format (ImageIO returned null). Consider external pre-conversion.")
            );
        }

        ImageAnalysis analysis = analyzeImage(image);

        ImageRecord record = new ImageRecord();
        record.imageName = fileName;
        record.folderName = folderName;
        record.flooding = floodFlag;
        record.season = inferSeasonFromFilename(fileName);
        if (record.season == null) record.season = "";

        String polarization = inferPolarizationFromFilename(fileName);
        record.polarization = (polarization != null) ? polarization : "";

        record.rawMean = analysis.rawMean;
        record.rawMeanNoZero = analysis.rawMeanNoZero;
        record.blackShape = analysis.blackShape;
        record.blackWidth = analysis.blackWidth;
        record.blackHeight = analysis.blackHeight;
        record.blackDiameter = analysis.blackDiameter;
        record.whiteShape = analysis.whiteShape;
        record.whiteWidth = analysis.whiteWidth;
        record.whiteHeight = analysis.whiteHeight;
        record.whiteDiameter = analysis.whiteDiameter;
        record.dominantShape = analysis.dominantShape;
        record.rawCounts = analysis.rawCounts;

        return new JobResult(record, null);
    }

    private static String inferSeasonFromFilename(String filename) {
        // Prefer an in-string YYYYMMDD (e.g., 20190118) rather than the first digits.
        java.util.regex.Matcher matcher = java.util.regex.Pattern.compile("20\\d{6}").matcher(filename);
        while (matcher.find()) {
            String yyyymmdd = matcher.group();
            int month;
            try {
                month = Integer.parseInt(yyyymmdd.substring(4, 6));
            } catch (NumberFormatException e) {
                continue;
            }
            if (month < 1 || month > 12) continue;

            if (month == 12 || month <= 2) return "Winter";
            if (month <= 5) return "Spring";
            if (month <= 8) return "Summer";
            if (month <= 11) return "Fall";
        }

        // Fallback to original digit-scan if no embedded date was found.
        String digits = filename.replaceAll("\\D+", "");
        if (digits.length() < 8) return null;

        String yyyymmdd = digits.substring(0, 8);
        int month;
        try {
            month = Integer.parseInt(yyyymmdd.substring(4, 6));
        } catch (NumberFormatException e) {
            return null;
        }

        if (month == 12 || month <= 2) return "Winter";
        if (month <= 5) return "Spring";
        if (month <= 8) return "Summer";
        if (month <= 11) return "Fall";
        return null;
    }

    private static String inferPolarizationFromFilename(String filename) {
        String upper = filename.toUpperCase(Locale.ROOT);
        if (upper.contains("_VV")) return "VV";
        if (upper.contains("_VH")) return "VH";
        return null;
    }

    // ---------- Image analysis (per image) ----------

    private static ImageAnalysis analyzeImage(BufferedImage image) {
        Raster raster = image.getRaster();
        int width = raster.getWidth();
        int height = raster.getHeight();
        int bands = raster.getNumBands();
        int transferType = raster.getTransferType();

        int maxSample;
        switch (transferType) {
            case DataBuffer.TYPE_BYTE:
                maxSample = 255;
                break;
            case DataBuffer.TYPE_USHORT:
                maxSample = 65535;
                break;
            default:
                maxSample = 65535;
                break;
        }

        int[] hist = new int[maxSample + 1];
        long sumRaw = 0L;
        long pixelCount = 0L;

        // RAW_00000 is treated as "null" for stats: we still count it in hist/rawCounts
        // but *exclude* it from sumRaw/pixelCount so it does not affect rawMean/std/etc.
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int raw = sampleRaw(raster, bands, maxSample, x, y);
                hist[raw]++;
                if (raw != 0) {          // <-- ignore RAW_00000 for statistical mean
                    sumRaw += raw;
                    pixelCount++;
                }
            }
        }

        double rawMean = (pixelCount > 0) ? (double) sumRaw / (double) pixelCount : 0.0;

        List<Integer> darkSeeds = new ArrayList<>();
        List<Integer> brightSeeds = new ArrayList<>();
        for (int v = 1; v <= maxSample && darkSeeds.size() < 3; v++) {
            if (hist[v] > 0) darkSeeds.add(v);
        }
        for (int v = maxSample; v >= 1 && brightSeeds.size() < 3; v--) {
            if (hist[v] > 0) brightSeeds.add(v);
        }

        Set<Integer> allowedBlack = buildAllowedShadeSet(darkSeeds, maxSample, 1);
        Set<Integer> allowedWhite = buildAllowedShadeSet(brightSeeds, maxSample, 1);

        Map<String, Integer> rawCounts = new HashMap<>();
        MasksAndComponents mac = buildMasksAndComponents(raster, bands, maxSample, width, height,
                allowedBlack, allowedWhite, rawCounts);

        double minAreaRatio = 0.05;
        double maxAreaRatio = 0.40;
        int totalPixels = width * height;

        if (mac.blackComponent.pixelCount < totalPixels * minAreaRatio ||
                mac.whiteComponent.pixelCount < totalPixels * minAreaRatio) {
            // Expand tolerance to ±2 if initial contiguous region is too small.
            allowedBlack = buildAllowedShadeSet(darkSeeds, maxSample, 2);
            allowedWhite = buildAllowedShadeSet(brightSeeds, maxSample, 2);
            mac = buildMasksAndComponents(raster, bands, maxSample, width, height, allowedBlack, allowedWhite, rawCounts);
        }

        Component blackComponent = mac.blackComponent;
        Component whiteComponent = mac.whiteComponent;

        int maxPixels = (int) Math.max(1, maxAreaRatio * totalPixels);
        if (blackComponent.pixelCount > maxPixels) blackComponent.pixelCount = 0;
        if (whiteComponent.pixelCount > maxPixels) whiteComponent.pixelCount = 0;

        if (blackComponent.pixelCount == 0) {
            blackComponent = new Component();
            blackComponent.update(width / 2, height / 2);
        }
        if (whiteComponent.pixelCount == 0) {
            whiteComponent = new Component();
            whiteComponent.update(width / 2, height / 2);
        }

        if (blackComponent.pixelCount == 0 && whiteComponent.pixelCount > 0) {
            blackComponent.minX = whiteComponent.minX;
            blackComponent.maxX = whiteComponent.maxX;
            blackComponent.minY = whiteComponent.minY;
            blackComponent.maxY = whiteComponent.maxY;
            blackComponent.pixelCount = whiteComponent.pixelCount;
        } else if (whiteComponent.pixelCount == 0 && blackComponent.pixelCount > 0) {
            whiteComponent.minX = blackComponent.minX;
            whiteComponent.maxX = blackComponent.maxX;
            whiteComponent.minY = blackComponent.minY;
            whiteComponent.maxY = blackComponent.maxY;
            whiteComponent.pixelCount = blackComponent.pixelCount;
        } else if (blackComponent.pixelCount == 0 && whiteComponent.pixelCount == 0) {
            blackComponent.minX = 0;
            blackComponent.maxX = Math.max(0, width - 1);
            blackComponent.minY = 0;
            blackComponent.maxY = Math.max(0, height - 1);
            blackComponent.pixelCount = width * height;

            whiteComponent.minX = 0;
            whiteComponent.maxX = Math.max(0, width - 1);
            whiteComponent.minY = 0;
            whiteComponent.maxY = Math.max(0, height - 1);
            whiteComponent.pixelCount = width * height;
        }

        int halfWidth = width / 2;
        int halfHeight = height / 2;

        Component blackAdjusted = new Component();
        Component whiteAdjusted = new Component();

        boolean[][] blackMask = mac.blackMask;
        boolean[][] whiteMask = mac.whiteMask;

        for (int y = 0; y < height; y++) {
            boolean[] blackRow = blackMask[y];
            boolean[] whiteRow = whiteMask[y];

            for (int x = 0; x < width; x++) {
                boolean isBlack = blackRow[x];
                boolean isWhite = whiteRow[x];

                boolean blackAllowed =
                        !(x > halfWidth && y < halfHeight) &&
                                !(x > halfWidth && y > halfHeight);
                boolean whiteAllowed =
                        !(x < halfWidth && y < halfHeight) &&
                                !(x < halfWidth && y > halfHeight);

                if (isBlack && blackAllowed && !isWhite) {
                    blackAdjusted.update(x, y);
                }
                if (isWhite && whiteAllowed && !isBlack) {
                    whiteAdjusted.update(x, y);
                }
            }
        }

        if (blackAdjusted.pixelCount > 0) {
            blackComponent = blackAdjusted;
        }
        if (whiteAdjusted.pixelCount > 0) {
            whiteComponent = whiteAdjusted;
        }

        String blackShape = classifyShape(blackComponent);
        String whiteShape = classifyShape(whiteComponent);

        List<Map.Entry<String, Integer>> sortedRaw = new ArrayList<>(rawCounts.entrySet());
        sortedRaw.sort(new Comparator<Map.Entry<String, Integer>>() {
            @Override
            public int compare(Map.Entry<String, Integer> a, Map.Entry<String, Integer> b) {
                return Integer.compare(b.getValue(), a.getValue());
            }
        });

        String dominantShape = "none";
        if (!sortedRaw.isEmpty()) {
            String firstKey = sortedRaw.get(0).getKey();
            try {
                int dominantRaw = Integer.parseInt(firstKey.substring(4));
                if (allowedBlack.contains(dominantRaw) && !allowedWhite.contains(dominantRaw)) {
                    dominantShape = "black";
                } else if (allowedWhite.contains(dominantRaw) && !allowedBlack.contains(dominantRaw)) {
                    dominantShape = "white";
                } else {
                    dominantShape = "mixed";
                }
            } catch (NumberFormatException e) {
                dominantShape = "mixed";
            }
        }

        ImageAnalysis analysis = new ImageAnalysis();
        analysis.rawMean = rawMean;
        analysis.rawMeanNoZero = rawMean;
        analysis.blackShape = blackShape;
        analysis.blackWidth = blackComponent.getWidth();
        analysis.blackHeight = blackComponent.getHeight();
        analysis.blackDiameter = blackComponent.getDiameter();
        analysis.whiteShape = whiteShape;
        analysis.whiteWidth = whiteComponent.getWidth();
        analysis.whiteHeight = whiteComponent.getHeight();
        analysis.whiteDiameter = whiteComponent.getDiameter();
        analysis.rawCounts = rawCounts;
        analysis.dominantShape = dominantShape;

        return analysis;
    }

    private static Set<Integer> buildAllowedShadeSet(List<Integer> seeds, int maxSample, int radius) {
        Set<Integer> allowed = new HashSet<>();
        for (int seed : seeds) {
            for (int delta = -radius; delta <= radius; delta++) {
                int v = seed + delta;
                if (v <= 0 || v > maxSample) continue;
                allowed.add(v);
            }
        }
        return allowed;
    }

    private static MasksAndComponents buildMasksAndComponents(
            Raster raster,
            int bands,
            int maxSample,
            int width,
            int height,
            Set<Integer> allowedBlack,
            Set<Integer> allowedWhite,
            Map<String, Integer> rawCounts
    ) {
        boolean[][] blackMask = new boolean[height][width];
        boolean[][] whiteMask = new boolean[height][width];

        for (int y = 0; y < height; y++) {
            boolean[] blackRow = blackMask[y];
            boolean[] whiteRow = whiteMask[y];
            for (int x = 0; x < width; x++) {
                int raw = sampleRaw(raster, bands, maxSample, x, y);
                String key = String.format("RAW_%05d", raw);
                rawCounts.put(key, rawCounts.getOrDefault(key, 0) + 1);

                boolean isBlack = allowedBlack.contains(raw);
                boolean isWhite = allowedWhite.contains(raw);

                if (isBlack && !isWhite) {
                    blackRow[x] = true;
                } else if (isWhite && !isBlack) {
                    whiteRow[x] = true;
                }
            }
        }

        int maxPixels = width * height;
        Component blackComponent = findLargestComponent(blackMask, maxPixels);
        Component whiteComponent = findLargestComponent(whiteMask, maxPixels);

        return new MasksAndComponents(blackComponent, whiteComponent, blackMask, whiteMask);
    }

    private static int sampleRaw(Raster raster, int bands, int maxSample, int x, int y) {
        int raw;
        if (bands == 1) {
            raw = raster.getSample(x, y, 0);
        } else if (bands >= 3) {
            int rSample = raster.getSample(x, y, 0);
            int gSample = raster.getSample(x, y, 1);
            int bSample = raster.getSample(x, y, 2);
            raw = (rSample + gSample + bSample) / 3;
        } else {
            raw = raster.getSample(x, y, 0);
        }
        if (raw < 0) raw = 0;
        if (raw > maxSample) raw = maxSample;
        return raw;
    }

    private static Component findLargestComponent(boolean[][] mask, int maxAllowedPixels) {
        int height = mask.length;
        int width = (height > 0) ? mask[0].length : 0;

        boolean[][] visited = new boolean[height][width];
        Component best = new Component();

        int[] dx = { 1, -1, 0, 0 };
        int[] dy = { 0, 0, 1, -1 };

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (!mask[y][x] || visited[y][x]) continue;

                Component comp = new Component();
                Deque<int[]> stack = new ArrayDeque<>();
                stack.push(new int[]{ x, y });
                visited[y][x] = true;

                while (!stack.isEmpty()) {
                    int[] p = stack.pop();
                    int cx = p[0];
                    int cy = p[1];

                    comp.update(cx, cy);

                    for (int k = 0; k < 4; k++) {
                        int nx = cx + dx[k];
                        int ny = cy + dy[k];

                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                        if (visited[ny][nx]) continue;
                        if (!mask[ny][nx]) continue;

                        visited[ny][nx] = true;
                        stack.push(new int[]{ nx, ny });
                    }
                }

                if (comp.pixelCount > best.pixelCount && comp.pixelCount <= maxAllowedPixels) {
                    best = comp;
                }
            }
        }

        return best;
    }

    private static String classifyShape(Component comp) {
        if (comp.pixelCount == 0) return "none";

        int width = comp.getWidth();
        int height = comp.getHeight();
        if (width <= 0 || height <= 0) return "none";

        double aspect = (double) width / (double) height;
        if (aspect < 1.0) aspect = 1.0 / aspect;

        double fillRatio = (double) comp.pixelCount / (double) (width * height);

        if (fillRatio >= 0.88 && aspect <= 1.05) {
            return "square";
        }
        if (fillRatio >= 0.80 && aspect > 1.05 && aspect < 2.4) {
            return "rectangle";
        }
        if (fillRatio > 0.75 && aspect <= 1.10) {
            return "circle";
        }
        if (fillRatio > 0.62 && aspect > 1.10 && aspect < 2.1) {
            return "ellipse";
        }
        if (fillRatio > 0.45 && aspect >= 2.1 && aspect < 3.5) {
            return "parallelogram";
        }
        if (fillRatio > 0.32 && aspect >= 3.5) {
            return "trapezium";
        }
        if (fillRatio > 0.22) {
            return "triangle";
        }
        return "crescent";
    }

    // ---------- CSV builders ----------

    private static List<List<String>> buildImagesAllRows(List<ImageRecord> records, Set<String> allRawCodes) {
        List<List<String>> rows = new ArrayList<>();

        List<String> header = new ArrayList<>();
        header.add("image_name");
        header.add("folder_name");
        header.add("flooding");
        header.add("season");
        header.add("polarization");
        header.add("raw_mean");
        header.add("black_shape");
        header.add("black_width");
        header.add("black_height");
        header.add("black_diameter");
        header.add("white_shape");
        header.add("white_width");
        header.add("white_height");
        header.add("white_diameter");
        header.add("dominant_shape");
        header.addAll(allRawCodes);

        rows.add(header);

        for (ImageRecord rec : records) {
            List<String> row = new ArrayList<>();
            row.add(rec.imageName);
            row.add(rec.folderName);
            row.add(Boolean.toString(rec.flooding));
            row.add(rec.season);
            row.add(rec.polarization);
            row.add(Double.toString(rec.rawMean));
            row.add(rec.blackShape);
            row.add(Integer.toString(rec.blackWidth));
            row.add(Integer.toString(rec.blackHeight));
            row.add(Double.toString(rec.blackDiameter));
            row.add(rec.whiteShape);
            row.add(Integer.toString(rec.whiteWidth));
            row.add(Integer.toString(rec.whiteHeight));
            row.add(Double.toString(rec.whiteDiameter));
            row.add(rec.dominantShape);

            for (String rawCode : allRawCodes) {
                Integer count = rec.rawCounts.get(rawCode);
                row.add((count != null) ? Integer.toString(count) : "0");
            }

            rows.add(row);
        }

        return rows;
    }

    private static List<List<String>> buildSummaryAllRows(List<ImageRecord> records, LogisticBundle logisticBundle,
                                                          WeightContext weightContext) {
        List<List<String>> out = new ArrayList<>();

        // Descriptive header:
        List<String> header = new ArrayList<>();
        header.add("section");
        header.add("metric_name");
        header.add("value_a");
        header.add("value_b");
        header.add("value_c");
        header.add("value_d");
        header.add("notes");
        header.add("baseline_category");
        header.add("confidence_from_n");
        header.add("standard_error");
        header.add("ci_low_95");
        header.add("ci_high_95");
        header.add("margin_of_error_95");
        out.add(header);

        out.add(row7("NOTE", "columns_seasons_shapes",
                "SEASONS/SHAPES: value_a = flooding=true count, value_b = flooding=false count, value_c = true_rate",
                "",
                "",
                "",
                ""));
        out.add(row7("NOTE", "columns_weights",
                "WEIGHTS numeric columns show Cohen's d, overall mean, overall std, and samples;"
                        + " categorical columns show shrunken rate delta, empirical rate, samples, and reliability (n/(n+50))",
                "",
                "",
                "",
                ""));
        out.add(row7("NOTE", "columns_xy_decision",
                "XY_TABLE provides its own column guide; DECISION_RULE uses the notes column for the formula",
                "",
                "",
                "",
                ""));
        out.add(row7("NOTE", "layout",
                "Rows start with LOGIT_SUMMARY (formerly Summary_Updated_Java), then blank lines, then STATS/SEASONS/SHAPES/WEIGHTS/XY_TABLE/DECISION_RULE.",
                "",
                "",
                "",
                ""));
        out.add(row7("NOTE", "decision_table_guide",
                "Decision_Table_Java.csv lists all season/polarization/shape combinations with confidence labels; unstable rows include a note explaining why.",
                "",
                "",
                "",
                ""));

        if (records.isEmpty()) {
            out.add(row7("NOTE", "no_data", "", "", "", "",
                    "No images produced data rows; summary is empty."));
            return out;
        }

        if (logisticBundle != null && logisticBundle.stats != null && !logisticBundle.stats.isEmpty()) {
            out.add(row13("LOGIT_SUMMARY", "columns",
                    "empirical_flood_rate",
                    "logit_coefficient",
                    "odds_ratio",
                    "samples",
                    "baseline_category",
                    "confidence_from_n",
                    "standard_error",
                    "ci_low_95",
                    "ci_high_95",
                    "margin_of_error_95",
                    ""));

            for (CategoryStats cs : logisticBundle.stats) {
                out.add(row13("LOGIT_SUMMARY", cs.attribute + ":" + cs.category,
                        Double.toString(cs.empiricalFloodRate),
                        Double.toString(cs.logitCoefficient),
                        Double.toString(cs.oddsRatio),
                        Integer.toString(cs.samples),
                        cs.baselineCategory,
                        cs.confidenceFromN,
                        Double.toString(cs.standardError),
                        Double.toString(cs.ciLow95),
                        Double.toString(cs.ciHigh95),
                        Double.toString(cs.marginOfError95),
                        ""));
            }

            out.add(row7("", "", "", "", "", "", ""));
            out.add(row7("", "", "", "", "", "", ""));
        }

        out.addAll(buildStatsSection(records));
        out.addAll(buildSeasonsSection(records));
        out.addAll(buildShapesSection(records));
        out.addAll(buildWeightsSection(records, weightContext));
        out.addAll(buildXYTableSection(records));

        List<String> ruleRow = new ArrayList<>();
        ruleRow.add("DECISION_RULE");
        ruleRow.add("score_formula");
        ruleRow.add("");
        ruleRow.add("");
        ruleRow.add("");
        ruleRow.add("");
        ruleRow.add("score = confidence_adjusted(w_raw*z_raw_mean + w_bd*z_black_diameter + w_wd*z_white_diameter + w_season + w_pol + w_black_shape + w_white_shape); each categorical weight is multiplied by reliability n/(n+50); z_feature = (x - mean_all)/std_all; P(FLOODING=true) = 1/(1+exp(-score)).");
        out.add(ruleRow);

        return out;
    }

    // ---------- STATS section (post-mode from RAW pixel counts, skipping RAW_00000) ----------

    private static List<List<String>> buildStatsSection(List<ImageRecord> records) {
        List<List<String>> out = new ArrayList<>();

        List<Double> valsTrueAll = new ArrayList<>();
        List<Double> valsFalseAll = new ArrayList<>();

        for (ImageRecord rec : records) {
            if (rec.flooding) valsTrueAll.add(rec.rawMean);
            else valsFalseAll.add(rec.rawMean);
        }

        int nTrueAll = valsTrueAll.size();
        int nFalseAll = valsFalseAll.size();

        double meanTrueAll = mean(valsTrueAll);
        double meanFalseAll = mean(valsFalseAll);
        double stdTrueAll = stddev(valsTrueAll, meanTrueAll);
        double stdFalseAll = stddev(valsFalseAll, meanFalseAll);

        // 3-sigma outlier removal (per group) for post-mean stats
        List<Double> valsTrueNo = filterByZScore(valsTrueAll, meanTrueAll, stdTrueAll, 3.0);
        List<Double> valsFalseNo = filterByZScore(valsFalseAll, meanFalseAll, stdFalseAll, 3.0);

        int nTrueNo = valsTrueNo.size();
        int nFalseNo = valsFalseNo.size();

        double meanTrueNo = mean(valsTrueNo);
        double meanFalseNo = mean(valsFalseNo);
        double stdTrueNo = stddev(valsTrueNo, meanTrueNo);
        double stdFalseNo = stddev(valsFalseNo, meanFalseNo);

        // For median, ignore raw_mean == 0.0 after outlier removal so background-only cases do not dominate.
        List<Double> valsTrueNoNoZero = removeZeros(valsTrueNo);
        List<Double> valsFalseNoNoZero = removeZeros(valsFalseNo);

        double medianTrueNo = median(valsTrueNoNoZero);
        double medianFalseNo = median(valsFalseNoNoZero);

        double cohensDNo = cohenD(meanTrueNo, stdTrueNo, nTrueNo, meanFalseNo, stdFalseNo, nFalseNo);

        String medianTrueStr = valsTrueNoNoZero.isEmpty() ? "" : Double.toString(medianTrueNo);
        String medianFalseStr = valsFalseNoNoZero.isEmpty() ? "" : Double.toString(medianFalseNo);

        // Compute mode from RAW pixel counts across post-mean images, skipping RAW_00000 entirely.
        Map<Integer, Long> rawCountsTrue = new HashMap<>();
        Map<Integer, Long> rawCountsFalse = new HashMap<>();

        double thrTrue = (stdTrueAll == 0.0) ? Double.POSITIVE_INFINITY : 3.0 * stdTrueAll;
        double thrFalse = (stdFalseAll == 0.0) ? Double.POSITIVE_INFINITY : 3.0 * stdFalseAll;

        for (ImageRecord rec : records) {
            if (rec.flooding) {
                if (Math.abs(rec.rawMean - meanTrueAll) > thrTrue) {
                    continue; // outlier
                }
                accumulateRawCounts(rec, rawCountsTrue);
            } else {
                if (Math.abs(rec.rawMean - meanFalseAll) > thrFalse) {
                    continue; // outlier
                }
                accumulateRawCounts(rec, rawCountsFalse);
            }
        }

        Integer modeRawTrue = findModeRaw(rawCountsTrue, true);
        Integer modeRawFalse = findModeRaw(rawCountsFalse, true);

        String modeTrueStr = (modeRawTrue == null) ? "" : Integer.toString(modeRawTrue);
        String modeFalseStr = (modeRawFalse == null) ? "" : Integer.toString(modeRawFalse);

        out.add(row7("STATS", "count_images_all",
                Integer.toString(nTrueAll),
                Integer.toString(nFalseAll),
                "",
                "",
                "True/false image counts, pre-mean (no outlier removal)."));

        out.add(row7("STATS", "count_images_post_mean",
                Integer.toString(nTrueNo),
                Integer.toString(nFalseNo),
                "",
                "",
                "Images retained for post-mean (|x-mean| <= 3*std within each group)."));

        out.add(row7("STATS", "mean_raw_pre",
                Double.toString(meanTrueAll),
                Double.toString(meanFalseAll),
                "",
                "",
                "Pre-mean raw_mean across all images (zeros included)."));

        out.add(row7("STATS", "std_raw_pre",
                Double.toString(stdTrueAll),
                Double.toString(stdFalseAll),
                "",
                "",
                "Pre-std raw_mean across all images."));

        out.add(row7("STATS", "post_mean_raw",
                Double.toString(meanTrueNo),
                Double.toString(meanFalseNo),
                "",
                "",
                "Post-mean raw_mean after 3-sigma outlier removal (zeros included)."));

        out.add(row7("STATS", "post_std_raw",
                Double.toString(stdTrueNo),
                Double.toString(stdFalseNo),
                "",
                "",
                "Post-std raw_mean after 3-sigma outlier removal."));

        out.add(row7("STATS", "post_median_raw",
                medianTrueStr,
                medianFalseStr,
                "",
                "",
                "Post-median raw_mean after 3-sigma removal; zeros excluded from median calculation."));

        out.add(row7("STATS", "post_mode_raw",
                modeTrueStr,
                modeFalseStr,
                "",
                "",
                "Post-mode RAW value: RAW sample whose total pixel count is highest across post-mean images in each group; zero-count RAW values are ignored and RAW=0 is skipped."));

        out.add(row7("STATS", "cohens_d_post_mean_raw",
                Double.toString(cohensDNo),
                "",
                "",
                "",
                "Effect size on post-mean data: (mean_true - mean_false) / pooled_std."));

        return out;
    }

    private static void accumulateRawCounts(ImageRecord rec, Map<Integer, Long> accumulator) {
        for (Map.Entry<String, Integer> e : rec.rawCounts.entrySet()) {
            String key = e.getKey();
            if (!key.startsWith("RAW_")) continue;
            String numStr = key.substring(4);
            int rawVal;
            try {
                rawVal = Integer.parseInt(numStr);
            } catch (NumberFormatException ex) {
                continue;
            }
            if (rawVal == 0) continue; // skip RAW_00000 entirely for stats
            int count = e.getValue();
            if (count <= 0) continue;
            long current = accumulator.getOrDefault(rawVal, 0L);
            accumulator.put(rawVal, current + count);
        }
    }

    private static Integer findModeRaw(Map<Integer, Long> counts, boolean disallowZeroRaw) {
        Integer bestRaw = null;
        long bestCount = 0L;
        for (Map.Entry<Integer, Long> e : counts.entrySet()) {
            int raw = e.getKey();
            long c = e.getValue();
            if (c <= 0L) continue;
            if (disallowZeroRaw && raw == 0) continue; // skip RAW=0 when requested
            if (bestRaw == null || c > bestCount || (c == bestCount && raw < bestRaw)) {
                bestCount = c;
                bestRaw = raw;
            }
        }
        return bestRaw;
    }

    private static List<String> row13(String section, String name,
                                      String v1, String v2, String v3, String v4, String notes,
                                      String baseline, String confidence, String se,
                                      String ciLow, String ciHigh, String margin95) {
        List<String> r = new ArrayList<>();
        r.add(section);
        r.add(name);
        r.add(v1 == null ? "" : v1);
        r.add(v2 == null ? "" : v2);
        r.add(v3 == null ? "" : v3);
        r.add(v4 == null ? "" : v4);
        r.add(notes == null ? "" : notes);
        r.add(baseline == null ? "" : baseline);
        r.add(confidence == null ? "" : confidence);
        r.add(se == null ? "" : se);
        r.add(ciLow == null ? "" : ciLow);
        r.add(ciHigh == null ? "" : ciHigh);
        r.add(margin95 == null ? "" : margin95);
        return r;
    }

    private static List<String> row7(String section, String name,
                                     String v1, String v2, String v3, String v4, String notes) {
        return row13(section, name, v1, v2, v3, v4, notes, "", "", "", "", "", "");
    }

    // ---------- SEASONS section ----------

    private static List<List<String>> buildSeasonsSection(List<ImageRecord> records) {
        List<List<String>> out = new ArrayList<>();

        Map<String, Integer> trueCounts = new HashMap<>();
        Map<String, Integer> falseCounts = new HashMap<>();

        for (ImageRecord rec : records) {
            String season = (rec.season == null || rec.season.isEmpty()) ? "Unknown" : rec.season;
            if (rec.flooding) {
                trueCounts.put(season, trueCounts.getOrDefault(season, 0) + 1);
            } else {
                falseCounts.put(season, falseCounts.getOrDefault(season, 0) + 1);
            }
        }

        String[] seasons = new String[]{"Winter", "Spring", "Summer", "Autumn", "Unknown"};
        for (String s : seasons) {
            int ct = trueCounts.getOrDefault(s, 0);
            int cf = falseCounts.getOrDefault(s, 0);
            int total = ct + cf;
            double rate = (total > 0) ? ((double) ct / (double) total) : 0.0;
            out.add(row7("SEASONS", s,
                    Integer.toString(ct),
                    Integer.toString(cf),
                    Double.toString(rate),
                    "",
                    "True_rate = count_true / (count_true + count_false) for this season."));
        }

        return out;
    }

    // ---------- SHAPES section ----------

    private static List<List<String>> buildShapesSection(List<ImageRecord> records) {
        List<List<String>> out = new ArrayList<>();

        Map<String, Integer> blackTrue = new HashMap<>();
        Map<String, Integer> blackFalse = new HashMap<>();
        Map<String, Integer> whiteTrue = new HashMap<>();
        Map<String, Integer> whiteFalse = new HashMap<>();

        for (ImageRecord rec : records) {
            String bShape = (rec.blackShape == null || rec.blackShape.isEmpty()) ? "none" : rec.blackShape;
            String wShape = (rec.whiteShape == null || rec.whiteShape.isEmpty()) ? "none" : rec.whiteShape;
            if (rec.flooding) {
                blackTrue.put(bShape, blackTrue.getOrDefault(bShape, 0) + 1);
                whiteTrue.put(wShape, whiteTrue.getOrDefault(wShape, 0) + 1);
            } else {
                blackFalse.put(bShape, blackFalse.getOrDefault(bShape, 0) + 1);
                whiteFalse.put(wShape, whiteFalse.getOrDefault(wShape, 0) + 1);
            }
        }

        Set<String> allShapesBlack = new TreeSet<>(blackTrue.keySet());
        allShapesBlack.addAll(blackFalse.keySet());
        for (String s : allShapesBlack) {
            int ct = blackTrue.getOrDefault(s, 0);
            int cf = blackFalse.getOrDefault(s, 0);
            out.add(row7("SHAPES", "black_" + s,
                    Integer.toString(ct + cf),
                    Integer.toString(ct),
                    "",
                    "",
                    "count_flooding=" + ct + "; count_nonflood=" + cf));
        }

        Set<String> allShapesWhite = new TreeSet<>(whiteTrue.keySet());
        allShapesWhite.addAll(whiteFalse.keySet());
        for (String s : allShapesWhite) {
            int ct = whiteTrue.getOrDefault(s, 0);
            int cf = whiteFalse.getOrDefault(s, 0);
            out.add(row7("SHAPES", "white_" + s,
                    Integer.toString(ct + cf),
                    Integer.toString(ct),
                    "",
                    "",
                    "count_flooding=" + ct + "; count_nonflood=" + cf));
        }

        return out;
    }

    // ---------- WEIGHTS section (numeric + season + shape + polarization) ----------

    private static List<List<String>> buildWeightsSection(List<ImageRecord> records, WeightContext ctx) {
        List<List<String>> out = new ArrayList<>();

        out.add(row7("WEIGHTS", "numeric_columns",
                "value_a = Cohen's d",
                "value_b = overall mean",
                "value_c = overall std",
                "value_d = samples",
                "Numeric weights compare flooding=true vs flooding=false using Cohen's d."));

        out.add(row7("WEIGHTS", "raw_mean",
                Double.toString(ctx.wRaw),
                Double.toString(ctx.meanRawAll),
                Double.toString(ctx.stdRawAll),
                Integer.toString(ctx.sampleCount),
                "raw_mean is over non-zero pixels only."));

        out.add(row7("WEIGHTS", "black_diameter",
                Double.toString(ctx.wBd),
                Double.toString(ctx.meanBdAll),
                Double.toString(ctx.stdBdAll),
                Integer.toString(ctx.sampleCount),
                ""));

        out.add(row7("WEIGHTS", "white_diameter",
                Double.toString(ctx.wWd),
                Double.toString(ctx.meanWdAll),
                Double.toString(ctx.stdWdAll),
                Integer.toString(ctx.sampleCount),
                ""));

        out.add(row7("WEIGHTS", "categorical_columns",
                "value_a = shrunken (rate - overall_true_rate)",
                "value_b = empirical rate",
                "value_c = samples",
                "value_d = reliability (n/(n+50))",
                "True/false counts and shrinkage avoid over-weighting tiny groups."));

        for (Map.Entry<String, WeightEntry> e : ctx.seasonWeight.entrySet()) {
            String s = e.getKey();
            WeightEntry w = e.getValue();
            out.add(row7("WEIGHTS", "season_" + s,
                    Double.toString(w.weight),
                    Double.toString(w.rate),
                    Integer.toString(w.trueCount + w.falseCount),
                    Double.toString(w.reliability),
                    "true=" + w.trueCount + ", false=" + w.falseCount + "; overall_true_rate=" + ctx.overallTrueRate));
        }

        for (Map.Entry<String, WeightEntry> e : ctx.polWeight.entrySet()) {
            String pol = e.getKey();
            WeightEntry w = e.getValue();
            out.add(row7("WEIGHTS", "pol_" + pol,
                    Double.toString(w.weight),
                    Double.toString(w.rate),
                    Integer.toString(w.trueCount + w.falseCount),
                    Double.toString(w.reliability),
                    "true=" + w.trueCount + ", false=" + w.falseCount + "; overall_true_rate=" + ctx.overallTrueRate));
        }

        for (Map.Entry<String, WeightEntry> e : ctx.blackShapeWeight.entrySet()) {
            String s = e.getKey();
            WeightEntry w = e.getValue();
            out.add(row7("WEIGHTS", "black_shape_" + s,
                    Double.toString(w.weight),
                    Double.toString(w.rate),
                    Integer.toString(w.trueCount + w.falseCount),
                    Double.toString(w.reliability),
                    "true=" + w.trueCount + ", false=" + w.falseCount + "; overall_true_rate=" + ctx.overallTrueRate));
        }

        for (Map.Entry<String, WeightEntry> e : ctx.whiteShapeWeight.entrySet()) {
            String s = e.getKey();
            WeightEntry w = e.getValue();
            out.add(row7("WEIGHTS", "white_shape_" + s,
                    Double.toString(w.weight),
                    Double.toString(w.rate),
                    Integer.toString(w.trueCount + w.falseCount),
                    Double.toString(w.reliability),
                    "true=" + w.trueCount + ", false=" + w.falseCount + "; overall_true_rate=" + ctx.overallTrueRate));
        }

        return out;
    }

    private static class WeightEntry {
        double weight;
        double rate;
        int trueCount;
        int falseCount;
        double reliability;
    }

    private static class WeightContext {
        double overallTrueRate;
        double wRaw, meanRawAll, stdRawAll;
        double wBd, meanBdAll, stdBdAll;
        double wWd, meanWdAll, stdWdAll;
        int sampleCount;
        Map<String, WeightEntry> seasonWeight = new LinkedHashMap<>();
        Map<String, WeightEntry> polWeight = new LinkedHashMap<>();
        Map<String, WeightEntry> blackShapeWeight = new LinkedHashMap<>();
        Map<String, WeightEntry> whiteShapeWeight = new LinkedHashMap<>();
    }

    private static WeightContext computeWeights(List<ImageRecord> records) {
        WeightContext ctx = new WeightContext();

        int totalTrue = 0, totalFalse = 0;
        for (ImageRecord rec : records) {
            if (rec.flooding) totalTrue++;
            else totalFalse++;
        }
        ctx.overallTrueRate = (totalTrue + totalFalse > 0)
                ? ((double) totalTrue / (double) (totalTrue + totalFalse))
                : 0.0;
        ctx.sampleCount = records.size();

        List<Double> rawTrue = new ArrayList<>();
        List<Double> rawFalse = new ArrayList<>();
        List<Double> rawAll = new ArrayList<>();

        List<Double> bdTrue = new ArrayList<>();
        List<Double> bdFalse = new ArrayList<>();
        List<Double> bdAll = new ArrayList<>();

        List<Double> wdTrue = new ArrayList<>();
        List<Double> wdFalse = new ArrayList<>();
        List<Double> wdAll = new ArrayList<>();

        for (ImageRecord rec : records) {
            double rm = rec.rawMean;
            double bd = rec.blackDiameter;
            double wd = rec.whiteDiameter;

            rawAll.add(rm);
            bdAll.add(bd);
            wdAll.add(wd);

            if (rec.flooding) {
                rawTrue.add(rm);
                bdTrue.add(bd);
                wdTrue.add(wd);
            } else {
                rawFalse.add(rm);
                bdFalse.add(bd);
                wdFalse.add(wd);
            }
        }

        ctx.meanRawAll = mean(rawAll);
        ctx.stdRawAll = stddev(rawAll, ctx.meanRawAll);
        double meanRawTrue = mean(rawTrue);
        double meanRawFalse = mean(rawFalse);
        double sdRawTrue = stddev(rawTrue, meanRawTrue);
        double sdRawFalse = stddev(rawFalse, meanRawFalse);
        ctx.wRaw = cohenD(meanRawTrue, sdRawTrue, rawTrue.size(),
                meanRawFalse, sdRawFalse, rawFalse.size());

        ctx.meanBdAll = mean(bdAll);
        ctx.stdBdAll = stddev(bdAll, ctx.meanBdAll);
        double meanBdTrue = mean(bdTrue);
        double meanBdFalse = mean(bdFalse);
        double sdBdTrue = stddev(bdTrue, meanBdTrue);
        double sdBdFalse = stddev(bdFalse, meanBdFalse);
        ctx.wBd = cohenD(meanBdTrue, sdBdTrue, bdTrue.size(),
                meanBdFalse, sdBdFalse, bdFalse.size());

        ctx.meanWdAll = mean(wdAll);
        ctx.stdWdAll = stddev(wdAll, ctx.meanWdAll);
        double meanWdTrue = mean(wdTrue);
        double meanWdFalse = mean(wdFalse);
        double sdWdTrue = stddev(wdTrue, meanWdTrue);
        double sdWdFalse = stddev(wdFalse, meanWdFalse);
        ctx.wWd = cohenD(meanWdTrue, sdWdTrue, wdTrue.size(),
                meanWdFalse, sdWdFalse, wdFalse.size());

        ctx.seasonWeight.putAll(computeCategoricalWeights(records, ctx.overallTrueRate,
                rec -> (rec.season == null || rec.season.isEmpty()) ? "Unknown" : rec.season));

        ctx.polWeight.putAll(computeCategoricalWeights(records, ctx.overallTrueRate,
                rec -> rec.polarization == null ? "OTHER" : rec.polarization));

        ctx.blackShapeWeight.putAll(computeCategoricalWeights(records, ctx.overallTrueRate,
                rec -> (rec.blackShape == null || rec.blackShape.isEmpty()) ? "none" : rec.blackShape));

        ctx.whiteShapeWeight.putAll(computeCategoricalWeights(records, ctx.overallTrueRate,
                rec -> (rec.whiteShape == null || rec.whiteShape.isEmpty()) ? "none" : rec.whiteShape));

        return ctx;
    }

    private static Map<String, WeightEntry> computeCategoricalWeights(
            List<ImageRecord> records,
            double overallTrueRate,
            java.util.function.Function<ImageRecord, String> classifier
    ) {
        Map<String, Integer> trueCounts = new HashMap<>();
        Map<String, Integer> falseCounts = new HashMap<>();

        for (ImageRecord rec : records) {
            String key = classifier.apply(rec);
            if (rec.flooding) {
                trueCounts.put(key, trueCounts.getOrDefault(key, 0) + 1);
            } else {
                falseCounts.put(key, falseCounts.getOrDefault(key, 0) + 1);
            }
        }

        Set<String> all = new TreeSet<>(trueCounts.keySet());
        all.addAll(falseCounts.keySet());

        Map<String, WeightEntry> out = new LinkedHashMap<>();
        for (String key : all) {
            int ct = trueCounts.getOrDefault(key, 0);
            int cf = falseCounts.getOrDefault(key, 0);
            int total = ct + cf;
            if (total == 0) continue;

            double rate = (double) ct / (double) total;
            double reliability = total / (total + 50.0); // shrink small or imbalanced groups
            double weight = (rate - overallTrueRate) * reliability;

            WeightEntry entry = new WeightEntry();
            entry.weight = weight;
            entry.rate = rate;
            entry.trueCount = ct;
            entry.falseCount = cf;
            entry.reliability = reliability;
            out.put(key, entry);
        }
        return out;
    }

    // ---------- XY_TABLE section (empirical probability by season/pol/shapes within |z_raw_mean| <= 1) ----------

    private static List<List<String>> buildXYTableSection(List<ImageRecord> records) {
        List<List<String>> out = new ArrayList<>();

        out.add(row7(
                "XY_TABLE", "COLUMNS",
                "total_images",
                "prob_true_percent",
                "count_true",
                "count_false",
                "For XY_TABLE rows: metric_name encodes season, polarization, black_shape, white_shape; value_a = total images in this combination (within |z_raw_mean| <= 1), value_b = P(FLOODING=true)%, value_c = number of true-labelled images, value_d = number of false-labelled images."));

        if (records.isEmpty()) {
            return out;
        }

        double sum = 0.0;
        double sumSq = 0.0;
        int n = 0;
        for (ImageRecord r : records) {
            double v = r.rawMean;
            if (Double.isNaN(v)) continue;
            sum += v;
            sumSq += v * v;
            n++;
        }
        if (n == 0) {
            return out;
        }
        double mean = sum / n;
        double var = (sumSq / n) - (mean * mean);
        double std = var > 0.0 ? Math.sqrt(var) : 0.0;
        if (std == 0.0) {
            return out;
        }

        double zThreshold = 1.0; // within one std dev of overall mean

        Map<String, int[]> comboCounts = new TreeMap<>();
        for (ImageRecord r : records) {
            double v = r.rawMean;
            if (Double.isNaN(v)) continue;
            double z = (v - mean) / std;
            if (Math.abs(z) > zThreshold) continue;

            String season = (r.season == null || r.season.isEmpty()) ? "Unknown" : r.season;
            String pol = (r.polarization == null) ? "" : r.polarization;

            String bs = (r.blackShape == null || r.blackShape.isEmpty())
                    ? "none" : r.blackShape;

            String ws = (r.whiteShape == null || r.whiteShape.isEmpty())
                    ? "none" : r.whiteShape;

            String key = "season=" + season + ",pol=" + pol +
                    ",black_shape=" + bs + ",white_shape=" + ws;

            int[] counts = comboCounts.get(key);
            if (counts == null) {
                counts = new int[]{0, 0}; // [false, true]
                comboCounts.put(key, counts);
            }
            if (r.flooding) {
                counts[1]++;
            } else {
                counts[0]++;
            }
        }

        int minComboCount = 10; // only show combinations with enough data
        for (Map.Entry<String, int[]> e : comboCounts.entrySet()) {
            String key = e.getKey();
            int[] counts = e.getValue();
            int cFalse = counts[0];
            int cTrue = counts[1];
            int total = cFalse + cTrue;
            if (total < minComboCount) continue;

            double prob = (total > 0) ? ((double) cTrue / (double) total) * 100.0 : 0.0;
            out.add(row7(
                    "XY_TABLE", key,
                    Integer.toString(total),
                    Double.toString(prob),
                    Integer.toString(cTrue),
                    Integer.toString(cFalse),
                    ""));
        }

        return out;
    }

    // ---------- basic stats helpers ----------

    private static double mean(List<Double> vals) {
        int n = vals.size();
        if (n == 0) return 0.0;
        double s = 0.0;
        for (double v : vals) s += v;
        return s / n;
    }

    private static double stddev(List<Double> vals, double mean) {
        int n = vals.size();
        if (n <= 1) return 0.0;
        double s2 = 0.0;
        for (double v : vals) {
            double d = v - mean;
            s2 += d * d;
        }
        return Math.sqrt(s2 / n);
    }

    private static List<Double> filterByZScore(List<Double> vals, double mean, double std, double z) {
        if (vals.isEmpty() || std == 0.0) return new ArrayList<>(vals);
        List<Double> out = new ArrayList<>();
        double thr = z * std;
        for (double v : vals) {
            if (Math.abs(v - mean) <= thr) out.add(v);
        }
        return out;
    }

    private static double median(List<Double> vals) {
        int n = vals.size();
        if (n == 0) return 0.0;
        List<Double> copy = new ArrayList<>(vals);
        Collections.sort(copy);
        if (n % 2 == 1) return copy.get(n / 2);
        return 0.5 * (copy.get(n / 2 - 1) + copy.get(n / 2));
    }

    private static List<Double> removeZeros(List<Double> vals) {
        List<Double> out = new ArrayList<>();
        for (double v : vals) {
            if (v != 0.0) {
                out.add(v);
            }
        }
        return out;
    }

    private static double cohenD(double mean1, double std1, int n1,
                                 double mean2, double std2, int n2) {
        if (n1 < 2 || n2 < 2) return 0.0;
        double var1 = std1 * std1;
        double var2 = std2 * std2;
        double pooled = Math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (double) (n1 + n2 - 2));
        if (pooled == 0.0) return 0.0;
        return (mean1 - mean2) / pooled;
    }
    private static List<List<String>> buildSkippedRows(List<SkipRecord> skips) {
        List<List<String>> rows = new ArrayList<>();
        rows.add(Arrays.asList("image_name", "folder_name", "reason"));
        for (SkipRecord s : skips) {
            rows.add(Arrays.asList(s.imageName, s.folderName, s.reason));
        }
        return rows;
    }

    // ---------- Logistic summaries (former SummaryGeneratorLogistic) ----------

    private static LogisticBundle computeLogisticBundle(List<ImageRecord> records) {
        if (records.isEmpty()) {
            return null;
        }

        String seasonBaseline = "Spring";
        String polBaseline = "VH";
        String blackBaseline = "none";
        String whiteBaseline = "none";

        LogisticModel model = buildAndTrainLogistic(records,
                seasonBaseline,
                polBaseline,
                blackBaseline,
                whiteBaseline);

        LogisticBundle bundle = new LogisticBundle();
        bundle.stats = new ArrayList<>();
        bundle.stats.addAll(computeCategoryStats(records, "season", seasonBaseline, model));
        bundle.stats.addAll(computeCategoryStats(records, "polarization", polBaseline, model));
        bundle.stats.addAll(computeCategoryStats(records, "black_shape", blackBaseline, model));
        bundle.stats.addAll(computeCategoryStats(records, "white_shape", whiteBaseline, model));
        bundle.combos = computeDecisionCombos(records);
        return bundle;
    }

    private static void writeLogisticSummaries(Path rootFolder, List<ImageRecord> records, LogisticBundle bundle) throws IOException {
        if (records.isEmpty()) {
            System.out.println("[WARN] Skipping logistic summaries because there are no image records.");
            return;
        }

        LogisticBundle usable = (bundle != null) ? bundle : computeLogisticBundle(records);
        if (usable == null) {
            System.out.println("[WARN] Skipping logistic summaries because bundle could not be computed.");
            return;
        }

        Path decisionPath = rootFolder.resolve("Decision_Table_Java.csv");
        writeDecisionTableCsv(usable.combos, decisionPath.toFile());

        System.out.println("[INFO] Decision table written to: " + decisionPath.toAbsolutePath());
    }

    private static LogisticModel buildAndTrainLogistic(List<ImageRecord> records,
                                                       String seasonBaseline,
                                                       String polBaseline,
                                                       String blackBaseline,
                                                       String whiteBaseline) {
        Set<String> seasonCats = new TreeSet<>();
        Set<String> polCats = new TreeSet<>();
        Set<String> bshapeCats = new TreeSet<>();
        Set<String> wshapeCats = new TreeSet<>();

        for (ImageRecord r : records) {
            seasonCats.add(safe(r.season));
            polCats.add(safe(r.polarization));
            bshapeCats.add(safe(r.blackShape));
            wshapeCats.add(safe(r.whiteShape));
        }

        Map<String, Integer> featIndex = new LinkedHashMap<>();
        int idx = 1; // 0 = bias

        idx = addDummyFeatures("season", seasonCats, seasonBaseline, featIndex, idx);
        idx = addDummyFeatures("polarization", polCats, polBaseline, featIndex, idx);
        idx = addDummyFeatures("black_shape", bshapeCats, blackBaseline, featIndex, idx);
        idx = addDummyFeatures("white_shape", wshapeCats, whiteBaseline, featIndex, idx);

        int rawMeanIndex = idx++;
        int blackDiamIndex = idx++;

        int numFeatures = idx;

        int n = records.size();
        double[][] X = new double[n][numFeatures];
        double[] y = new double[n];

        for (int i = 0; i < n; i++) {
            ImageRecord r = records.get(i);
            X[i][0] = 1.0; // bias

            setDummy(X[i], featIndex, "season", safe(r.season), seasonBaseline);
            setDummy(X[i], featIndex, "polarization", safe(r.polarization), polBaseline);
            setDummy(X[i], featIndex, "black_shape", safe(r.blackShape), blackBaseline);
            setDummy(X[i], featIndex, "white_shape", safe(r.whiteShape), whiteBaseline);

            X[i][rawMeanIndex] = r.rawMean;
            X[i][blackDiamIndex] = r.blackDiameter;

            y[i] = r.flooding ? 1.0 : 0.0;
        }

        standardizeFeatures(X);

        double[] w = trainLogistic(X, y);

        LogisticModel model = new LogisticModel();
        model.featureIndex = featIndex;
        model.rawMeanIndex = rawMeanIndex;
        model.blackDiameterIndex = blackDiamIndex;
        model.numFeatures = numFeatures;
        model.weights = w;
        return model;
    }

    private static int addDummyFeatures(String attrName,
                                        Set<String> cats,
                                        String baseline,
                                        Map<String, Integer> featIndex,
                                        int startIdx) {
        int idx = startIdx;
        for (String cat : cats) {
            String value = (cat == null) ? "" : cat;
            if (value.equals(baseline)) continue;
            String key = attrName + "=" + value;
            featIndex.put(key, idx++);
        }
        return idx;
    }

    private static void setDummy(double[] x,
                                 Map<String, Integer> featIndex,
                                 String attrName,
                                 String value,
                                 String baseline) {
        String val = (value == null) ? "" : value;
        if (val.equals(baseline)) return;
        String key = attrName + "=" + val;
        Integer idx = featIndex.get(key);
        if (idx != null) {
            x[idx] = 1.0;
        }
    }

    private static void standardizeFeatures(double[][] X) {
        int n = X.length;
        if (n == 0) return;
        int m = X[0].length;

        double[] mean = new double[m];
        double[] std = new double[m];

        for (int j = 1; j < m; j++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += X[i][j];
            }
            mean[j] = sum / n;

            double var = 0.0;
            for (int i = 0; i < n; i++) {
                double d = X[i][j] - mean[j];
                var += d * d;
            }
            var /= n;
            std[j] = Math.sqrt(var);
        }

        for (int j = 1; j < m; j++) {
            if (std[j] == 0.0) continue;
            for (int i = 0; i < n; i++) {
                X[i][j] = (X[i][j] - mean[j]) / std[j];
            }
        }
    }

    private static double[] trainLogistic(double[][] X, double[] y) {
        int n = X.length;
        int m = X[0].length;

        double[] w = new double[m];
        double learningRate = 0.1;
        double lambda = 0.01;
        int maxIter = 5000;

        for (int iter = 0; iter < maxIter; iter++) {
            double[] grad = new double[m];

            for (int i = 0; i < n; i++) {
                double z = 0.0;
                for (int j = 0; j < m; j++) {
                    z += w[j] * X[i][j];
                }
                double p = sigmoid(z);
                double error = p - y[i];

                for (int j = 0; j < m; j++) {
                    grad[j] += error * X[i][j];
                }
            }

            for (int j = 1; j < m; j++) {
                grad[j] += lambda * w[j];
            }

            for (int j = 0; j < m; j++) {
                w[j] -= learningRate * grad[j] / n;
            }
        }

        return w;
    }

    private static double sigmoid(double z) {
        if (z > 20) return 1.0;
        if (z < -20) return 0.0;
        return 1.0 / (1.0 + Math.exp(-z));
    }

    private static String confidenceLabel(int n) {
        if (n >= 500) return "High";
        if (n >= 100) return "Medium";
        if (n >= 20) return "Low";
        if (n >= 5) return "Very Low";
        return "Extremely Low";
    }

    private static List<CategoryStats> computeCategoryStats(List<ImageRecord> records,
                                                            String attributeName,
                                                            String baseline,
                                                            LogisticModel model) {
        Map<String, List<ImageRecord>> byCat = new HashMap<>();

        for (ImageRecord r : records) {
            String key;
            switch (attributeName) {
                case "season":
                    key = safe(r.season);
                    break;
                case "polarization":
                    key = safe(r.polarization);
                    break;
                case "black_shape":
                    key = safe(r.blackShape);
                    break;
                case "white_shape":
                    key = safe(r.whiteShape);
                    break;
                default:
                    continue;
            }
            byCat.computeIfAbsent(key, k -> new ArrayList<>()).add(r);
        }

        List<CategoryStats> result = new ArrayList<>();

        for (Map.Entry<String, List<ImageRecord>> e : byCat.entrySet()) {
            String cat = e.getKey();
            List<ImageRecord> list = e.getValue();
            int n = list.size();
            int k = 0;
            for (ImageRecord r : list) {
                if (r.flooding) k++;
            }

            double p = (n > 0) ? (k / (double) n) : Double.NaN;
            double se = Double.NaN;
            double ciLow = Double.NaN;
            double ciHigh = Double.NaN;
            double margin = Double.NaN;
            if (n > 0) {
                se = Math.sqrt(p * (1.0 - p) / n);
                ciLow = Math.max(0.0, p - 1.96 * se);
                ciHigh = Math.min(1.0, p + 1.96 * se);
                margin = 1.96 * se;
            }

            CategoryStats cs = new CategoryStats();
            cs.attribute = attributeName;
            cs.category = cat;
            cs.samples = n;
            cs.empiricalFloodRate = p;
            cs.standardError = se;
            cs.ciLow95 = ciLow;
            cs.ciHigh95 = ciHigh;
            cs.marginOfError95 = margin;
            cs.isBaseline = (baseline != null && baseline.equals(cat));
            cs.baselineCategory = cs.isBaseline ? baseline : "";
            cs.confidenceFromN = confidenceLabel(n);

            if (cs.isBaseline) {
                cs.logitCoefficient = 0.0;
                cs.oddsRatio = 1.0;
            } else {
                String key = attributeName + "=" + cat;
                Integer idx = model.featureIndex.get(key);
                if (idx == null) {
                    cs.logitCoefficient = 0.0;
                    cs.oddsRatio = 1.0;
                } else {
                    double beta = model.weights[idx];
                    cs.logitCoefficient = beta;
                    cs.oddsRatio = Math.exp(beta);
                }
            }

            result.add(cs);
        }

        result.sort(Comparator.comparing(cs -> cs.category));
        return result;
    }

    private static List<DecisionCombo> computeDecisionCombos(List<ImageRecord> records) {
        Map<String, DecisionCombo> map = new HashMap<>();

        for (ImageRecord r : records) {
            String key = comboKey(safe(r.season), safe(r.polarization), safe(r.blackShape), safe(r.whiteShape));
            DecisionCombo dc = map.get(key);
            if (dc == null) {
                dc = new DecisionCombo();
                dc.season = safe(r.season);
                dc.polarization = safe(r.polarization);
                dc.blackShape = safe(r.blackShape);
                dc.whiteShape = safe(r.whiteShape);
                dc.samples = 0;
                dc.floodCount = 0;
                map.put(key, dc);
            }
            dc.samples++;
            if (r.flooding) {
                dc.floodCount++;
            }
        }

        List<DecisionCombo> list = new ArrayList<>();
        for (DecisionCombo dc : map.values()) {
            if (dc.samples > 0) {
                dc.empiricalFloodRate = dc.floodCount / (double) dc.samples;
                double p = dc.empiricalFloodRate;
                double n = dc.samples;
                double se = Math.sqrt(p * (1.0 - p) / n);
                double moe = 1.96 * se;
                double ciLow = Math.max(0.0, p - 1.96 * se);
                double ciHigh = Math.min(1.0, p + 1.96 * se);
                dc.standardError = se;
                dc.marginOfError95 = moe;
                dc.ciLow95 = ciLow;
                dc.ciHigh95 = ciHigh;
            } else {
                dc.empiricalFloodRate = Double.NaN;
                dc.standardError = Double.NaN;
                dc.marginOfError95 = Double.NaN;
                dc.ciLow95 = Double.NaN;
                dc.ciHigh95 = Double.NaN;
            }
            dc.confidenceFromN = confidenceLabel(dc.samples);

            boolean extreme = (dc.empiricalFloodRate == 0.0 || dc.empiricalFloodRate == 1.0);
            boolean tiny = dc.samples < 10;
            boolean lowConf = dc.confidenceFromN.equals("Very Low") || dc.confidenceFromN.equals("Extremely Low");
            dc.unstableExtremeFlag = tiny || (extreme && lowConf);

            list.add(dc);
        }

        list.sort(Comparator.comparing((DecisionCombo dc) -> dc.empiricalFloodRate).reversed()
                .thenComparing(dc -> -dc.samples));
        return list;
    }

    private static String comboKey(String season, String pol, String bshape, String wshape) {
        return "s=" + season +
                "|p=" + pol +
                "|b=" + bshape +
                "|w=" + wshape;
    }

    private static void writeLogisticSummaryCsv(List<CategoryStats> stats, File outFile) throws IOException {
        try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(outFile)))) {
            pw.println(String.join(",",
                    "attribute",
                    "category",
                    "baseline_category",
                    "is_baseline",
                    "samples",
                    "empirical_flood_rate",
                    "standard_error",
                    "flood_rate_CI_low_95",
                    "flood_rate_CI_high_95",
                    "logit_coefficient",
                    "odds_ratio",
                    "margin_of_error_95",
                    "confidence_from_n"
            ));

            for (CategoryStats cs : stats) {
                pw.println(String.join(",",
                        cs.attribute,
                        cs.category,
                        cs.baselineCategory,
                        Boolean.toString(cs.isBaseline),
                        Integer.toString(cs.samples),
                        Double.toString(cs.empiricalFloodRate),
                        Double.toString(cs.standardError),
                        Double.toString(cs.ciLow95),
                        Double.toString(cs.ciHigh95),
                        Double.toString(cs.logitCoefficient),
                        Double.toString(cs.oddsRatio),
                        Double.toString(cs.marginOfError95),
                        cs.confidenceFromN
                ));
            }
        }
    }

    private static void writeDecisionTableCsv(List<DecisionCombo> combos, File outFile) throws IOException {
        try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(outFile)))) {
            pw.println(String.join(",",
                    "season",
                    "polarization",
                    "black_shape",
                    "white_shape",
                    "samples",
                    "empirical_flood_rate",
                    "standard_error",
                    "margin_of_error_95",
                    "flood_rate_CI_low_95",
                    "flood_rate_CI_high_95",
                    "confidence_from_n",
                    "unstable_extreme_flag",
                    "notes"
            ));

            combos.sort(Comparator.comparing((DecisionCombo dc) -> dc.empiricalFloodRate).reversed()
                    .thenComparing(dc -> dc.marginOfError95));

            for (DecisionCombo dc : combos) {
                String note = dc.unstableExtremeFlag
                        ? "unstable due to tiny sample or extreme rate; interpret cautiously"
                        : "";
                pw.println(String.join(",",
                        safe(dc.season),
                        safe(dc.polarization),
                        safe(dc.blackShape),
                        safe(dc.whiteShape),
                        Integer.toString(dc.samples),
                        Double.toString(dc.empiricalFloodRate),
                        Double.toString(dc.standardError),
                        Double.toString(dc.marginOfError95),
                        Double.toString(dc.ciLow95),
                        Double.toString(dc.ciHigh95),
                        dc.confidenceFromN,
                        Boolean.toString(dc.unstableExtremeFlag),
                        note
                ));
            }
        }
    }

    private static String safe(String s) {
        return (s == null) ? "" : s;
    }

    // ---------- CSV writing ----------

    private static void writeCsv(Path path, List<List<String>> rows) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            for (List<String> row : rows) {
                writer.write(escapeCsvRow(row));
                writer.newLine();
            }
        }
    }

    private static String escapeCsvRow(List<String> row) {
        StringBuilder sb = new StringBuilder();
        boolean first = true;
        for (String field : row) {
            if (!first) sb.append(',');
            first = false;
            if (field == null) field = "";
            boolean needQuotes =
                    field.contains(",") ||
                            field.contains("\"") ||
                            field.contains("\n") ||
                            field.contains("\r");
            if (needQuotes) {
                sb.append('"');
                for (int i = 0; i < field.length(); i++) {
                    char c = field.charAt(i);
                    if (c == '"') sb.append('"');
                    sb.append(c);
                }
                sb.append('"');
            } else {
                sb.append(field);
            }
        }
        return sb.toString();
    }

    // ---------- Auto post-processing / lightweight logistic-style scores ----------

    private static void writeAutoProbabilities(Path rootFolder, List<ImageRecord> records, WeightContext ctx) throws IOException {
        if (records.isEmpty()) {
            System.out.println("[WARN] Skipping Auto_Probabilities.csv because there are no image records.");
            return;
        }
        double confidenceAdjust = (ctx.sampleCount > 0)
                ? (ctx.sampleCount / (ctx.sampleCount + 50.0))
                : 0.0;

        Path autoPath = rootFolder.resolve("Auto_Probabilities.csv");

        try (BufferedWriter writer = Files.newBufferedWriter(autoPath, StandardCharsets.UTF_8)) {
            writer.write("# score = confidence_adjusted(w_raw*z_raw_mean + w_bd*z_black_diameter + w_wd*z_white_diameter + w_season + w_pol + w_black_shape + w_white_shape); probability = logistic(score); probability is the confidence-weighted chance that the image is flooded given the observed attributes.");
            writer.newLine();
            writer.write("image_name,folder_name,polarization,season,black_shape,white_shape,label_flooding,score,probability");
            writer.newLine();

            List<List<String>> rows = new ArrayList<>();

            for (ImageRecord rec : records) {
                double zRaw = (ctx.stdRawAll > 0.0) ? (rec.rawMean - ctx.meanRawAll) / ctx.stdRawAll : 0.0;
                double zBlack = (ctx.stdBdAll > 0.0) ? (rec.blackDiameter - ctx.meanBdAll) / ctx.stdBdAll : 0.0;
                double zWhite = (ctx.stdWdAll > 0.0) ? (rec.whiteDiameter - ctx.meanWdAll) / ctx.stdWdAll : 0.0;

                WeightEntry sw = ctx.seasonWeight.get(safe(rec.season));
                WeightEntry pw = ctx.polWeight.get(safe(rec.polarization));
                WeightEntry bw = ctx.blackShapeWeight.get(safe(rec.blackShape));
                WeightEntry ww = ctx.whiteShapeWeight.get(safe(rec.whiteShape));

                double scoreCore = (ctx.wRaw * zRaw)
                        + (ctx.wBd * zBlack)
                        + (ctx.wWd * zWhite)
                        + ((sw != null) ? sw.weight : 0.0)
                        + ((pw != null) ? pw.weight : 0.0)
                        + ((bw != null) ? bw.weight : 0.0)
                        + ((ww != null) ? ww.weight : 0.0);

                double score = scoreCore * confidenceAdjust;
                double probability = 1.0 / (1.0 + Math.exp(-score));

                List<String> row = Arrays.asList(
                        rec.imageName,
                        rec.folderName,
                        rec.polarization,
                        rec.season,
                        rec.blackShape,
                        rec.whiteShape,
                        Boolean.toString(rec.flooding),
                        Double.toString(score),
                        Double.toString(probability)
                );
                rows.add(row);
            }

            rows.sort((a, b) -> Double.compare(Double.parseDouble(b.get(8)), Double.parseDouble(a.get(8))));

            for (List<String> row : rows) {
                writer.write(escapeCsvRow(row));
                writer.newLine();
            }
        }

        System.out.println("[INFO] Auto_Probabilities.csv written with confidence-adjusted logistic scores in: " + rootFolder);
    }
}

