import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.DataBuffer;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.File;

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
            }
            if (Files.exists(s2JsonPath)) {
                parseFloodJsonFile(s2JsonPath, floodBySentinelName);
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

        // Gather all image jobs (root + numeric subfolders).
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

        // Build summary CSV sections.
        List<List<String>> summaryAllRows = buildSummaryAllRows(records);

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
            writeAutoProbabilities(rootFolder, records);
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

        try (DirectoryStream<Path> ds = Files.newDirectoryStream(rootFolder)) {
            for (Path entry : ds) {
                if (Files.isDirectory(entry)) {
                    String folderName = entry.getFileName().toString();
                    if (folderName.matches("\\d+")) {
                        count += gatherJobsInFolder(entry, folderName, jobs);
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("[WARN] Failed to scan subfolders: " + e.getMessage());
        }

        // Also process any TIF/TIFF in the root folder.
        count += gatherJobsInFolder(rootFolder, "", jobs);

        return count;
    }

    private static int gatherJobsInFolder(Path folder, String folderName, List<ImageJob> jobs) {
        int count = 0;

        try (DirectoryStream<Path> ds = Files.newDirectoryStream(folder)) {
            for (Path entry : ds) {
                if (Files.isRegularFile(entry)) {
                    String name = entry.getFileName().toString().toLowerCase(Locale.ROOT);
                    if (name.endsWith(".tif") || name.endsWith(".tiff")) {
                        jobs.add(new ImageJob(entry, folderName));
                        count++;
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("[WARN] Failed to list files in " + folder + ": " + e.getMessage());
        }

        return count;
    }

    // ---------- JSON parsing ----------

    private static void parseFloodJsonFile(Path jsonPath, Map<String, Boolean> floodBySentinelName) throws IOException {
        System.out.println("[INFO] Parsing JSON labels from: " + jsonPath);

        String json = Files.readString(jsonPath, StandardCharsets.UTF_8);

        // Match both FLOODING followed by filename and filename followed by FLOODING
        int before = floodBySentinelName.size();

        // Example pattern inside each object:
        //   "FLOODING": false ... "filename": "S2_2019-02-04"
        java.util.regex.Pattern floodingThenFile = java.util.regex.Pattern.compile(
                "\\\"FLOODING\\\"\\s*:\\s*(true|false).*?\\\"filename\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"",
                java.util.regex.Pattern.CASE_INSENSITIVE | java.util.regex.Pattern.DOTALL);

        // Sometimes filename may appear before FLOODING; match that as well.
        java.util.regex.Pattern fileThenFlooding = java.util.regex.Pattern.compile(
                "\\\"filename\\\"\\s*:\\s*\\\"([^\\\"]+)\\\".*?\\\"FLOODING\\\"\\s*:\\s*(true|false)",
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

        // Darkest 20 and brightest 20 present RAW values (zeros still included as "darkest").
        Set<Integer> darkSet = new HashSet<>();
        Set<Integer> brightSet = new HashSet<>();
        int darkNeeded = 20;
        int brightNeeded = 20;

        for (int v = 0; v <= maxSample && darkSet.size() < darkNeeded; v++) {
            if (hist[v] > 0) darkSet.add(v);
        }
        for (int v = maxSample; v >= 0 && brightSet.size() < brightNeeded; v--) {
            if (hist[v] > 0) brightSet.add(v);
        }

        Map<String, Integer> rawCounts = new HashMap<>();
        boolean[][] blackMask = new boolean[height][width];
        boolean[][] whiteMask = new boolean[height][width];

        for (int y = 0; y < height; y++) {
            boolean[] blackRow = blackMask[y];
            boolean[] whiteRow = whiteMask[y];

            for (int x = 0; x < width; x++) {
                int raw = sampleRaw(raster, bands, maxSample, x, y);
                String key = String.format("RAW_%05d", raw);
                int old = rawCounts.getOrDefault(key, 0);
                rawCounts.put(key, old + 1);

                boolean isBlack = darkSet.contains(raw);
                boolean isWhite = brightSet.contains(raw);

                if (isBlack && !isWhite) {
                    blackRow[x] = true;
                } else if (isWhite && !isBlack) {
                    whiteRow[x] = true;
                } else {
                    blackRow[x] = false;
                    whiteRow[x] = false;
                }
            }
        }

        Component blackComponent = findLargestComponent(blackMask);
        Component whiteComponent = findLargestComponent(whiteMask);

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
                if (darkSet.contains(dominantRaw) && !brightSet.contains(dominantRaw)) {
                    dominantShape = "black";
                } else if (brightSet.contains(dominantRaw) && !darkSet.contains(dominantRaw)) {
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

    private static Component findLargestComponent(boolean[][] mask) {
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

                if (comp.pixelCount > best.pixelCount) {
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

        if (fillRatio > 0.75 && aspect < 1.2) {
            return "square";
        }
        if (fillRatio > 0.75 && aspect >= 1.2 && aspect < 3.0) {
            return "rectangle";
        }
        if (fillRatio > 0.6 && aspect < 1.2) {
            return "circle";
        }
        if (fillRatio > 0.5 && aspect >= 1.2 && aspect < 2.5) {
            return "ellipse";
        }
        if (fillRatio > 0.4 && aspect >= 2.5) {
            return "parallelogram";
        }
        if (fillRatio > 0.3 && aspect >= 3.0) {
            return "trapezium";
        }
        if (fillRatio > 0.2) {
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

    private static List<List<String>> buildSummaryAllRows(List<ImageRecord> records) {
        List<List<String>> rows = new ArrayList<>();

        rows.add(Collections.singletonList("COUNTS"));
        rows.add(Arrays.asList("metric", "VV", "VH", "OTHER", "TOTAL"));

        Map<String, int[]> polCounts = new LinkedHashMap<>();
        polCounts.put("total_images", new int[4]);
        polCounts.put("flood_true", new int[4]);
        polCounts.put("flood_false", new int[4]);

        int idxVV = 0, idxVH = 1, idxOT = 2, idxTotal = 3;

        for (ImageRecord rec : records) {
            int idx;
            if ("VV".equalsIgnoreCase(rec.polarization)) {
                idx = idxVV;
            } else if ("VH".equalsIgnoreCase(rec.polarization)) {
                idx = idxVH;
            } else {
                idx = idxOT;
            }

            polCounts.get("total_images")[idx]++;
            polCounts.get("total_images")[idxTotal]++;

            if (rec.flooding) {
                polCounts.get("flood_true")[idx]++;
                polCounts.get("flood_true")[idxTotal]++;
            } else {
                polCounts.get("flood_false")[idx]++;
                polCounts.get("flood_false")[idxTotal]++;
            }
        }

        for (Map.Entry<String, int[]> e : polCounts.entrySet()) {
            String metric = e.getKey();
            int[] c = e.getValue();
            rows.add(Arrays.asList(
                    metric,
                    Integer.toString(c[idxVV]),
                    Integer.toString(c[idxVH]),
                    Integer.toString(c[idxOT]),
                    Integer.toString(c[idxTotal])
            ));
        }

        rows.add(Collections.emptyList());

        rows.add(Collections.singletonList("SEASONS"));
        rows.add(Arrays.asList("season", "count", "flood_true", "flood_false", "true_rate"));

        Map<String, int[]> seasonCounts = new LinkedHashMap<>();

        for (ImageRecord rec : records) {
            String season = (rec.season == null || rec.season.isEmpty()) ? "UNKNOWN" : rec.season;
            int[] counts = seasonCounts.computeIfAbsent(season, s -> new int[3]);
            counts[0]++;
            if (rec.flooding) {
                counts[1]++;
            } else {
                counts[2]++;
            }
        }

        for (Map.Entry<String, int[]> e : seasonCounts.entrySet()) {
            String season = e.getKey();
            int[] c = e.getValue();
            int count = c[0];
            int floodTrue = c[1];
            int floodFalse = c[2];
            double trueRate = (count > 0) ? (double) floodTrue / (double) count : 0.0;
            rows.add(Arrays.asList(
                    season,
                    Integer.toString(count),
                    Integer.toString(floodTrue),
                    Integer.toString(floodFalse),
                    Double.toString(trueRate)
            ));
        }

        rows.add(Collections.emptyList());

        rows.add(Collections.singletonList("SHAPES"));
        rows.add(Arrays.asList("shape_type", "shape_name", "count", "flood_true", "flood_false", "true_rate"));

        Map<String, Map<String, int[]>> shapeCounts = new LinkedHashMap<>();
        shapeCounts.put("black", new LinkedHashMap<>());
        shapeCounts.put("white", new LinkedHashMap<>());

        for (ImageRecord rec : records) {
            String bShape = (rec.blackShape == null || rec.blackShape.isEmpty()) ? "none" : rec.blackShape;
            String wShape = (rec.whiteShape == null || rec.whiteShape.isEmpty()) ? "none" : rec.whiteShape;

            int[] bCounts = shapeCounts.get("black").computeIfAbsent(bShape, s -> new int[3]);
            int[] wCounts = shapeCounts.get("white").computeIfAbsent(wShape, s -> new int[3]);

            bCounts[0]++;
            wCounts[0]++;

            if (rec.flooding) {
                bCounts[1]++;
                wCounts[1]++;
            } else {
                bCounts[2]++;
                wCounts[2]++;
            }
        }

        for (Map.Entry<String, Map<String, int[]>> eType : shapeCounts.entrySet()) {
            String type = eType.getKey();
            for (Map.Entry<String, int[]> eShape : eType.getValue().entrySet()) {
                String shape = eShape.getKey();
                int[] c = eShape.getValue();
                int count = c[0];
                int floodTrue = c[1];
                int floodFalse = c[2];
                double trueRate = (count > 0) ? (double) floodTrue / (double) count : 0.0;

                rows.add(Arrays.asList(
                        type,
                        shape,
                        Integer.toString(count),
                        Integer.toString(floodTrue),
                        Integer.toString(floodFalse),
                        Double.toString(trueRate)
                ));
            }
        }

        rows.add(Collections.emptyList());

        rows.add(Collections.singletonList("WEIGHTS"));
        rows.add(Arrays.asList("feature", "weight"));

        Map<Integer, Long> globalRawCounts = new TreeMap<>();
        for (ImageRecord rec : records) {
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
                int count = e.getValue();
                if (count <= 0) continue;
                long current = globalRawCounts.getOrDefault(rawVal, 0L);
                globalRawCounts.put(rawVal, current + count);
            }
        }

        rows.add(Arrays.asList("raw_mean", "1.0"));
        rows.add(Arrays.asList("black_diameter", "1.0"));
        rows.add(Arrays.asList("white_diameter", "1.0"));

        rows.add(Collections.emptyList());

        rows.add(Collections.singletonList("XY_TABLE"));
        rows.add(Arrays.asList("season", "polarization", "black_shape", "white_shape", "count", "flood_true", "true_rate"));

        Map<String, Map<String, Map<String, Map<String, int[]>>>> xy = new LinkedHashMap<>();

        List<Double> rawMeans = new ArrayList<>();
        for (ImageRecord rec : records) {
            rawMeans.add(rec.rawMean);
        }
        double overallMean = 0.0;
        for (double v : rawMeans) {
            overallMean += v;
        }
        overallMean = (rawMeans.isEmpty()) ? 0.0 : (overallMean / rawMeans.size());

        double overallVar = 0.0;
        for (double v : rawMeans) {
            double diff = v - overallMean;
            overallVar += diff * diff;
        }
        overallVar = (rawMeans.size() > 1) ? (overallVar / (rawMeans.size() - 1)) : 0.0;
        double overallStd = (overallVar > 0.0) ? Math.sqrt(overallVar) : 1.0;

        for (ImageRecord rec : records) {
            double z = (rec.rawMean - overallMean) / overallStd;
            if (Math.abs(z) > 1.0) continue;

            String season = (rec.season == null || rec.season.isEmpty()) ? "UNKNOWN" : rec.season;
            String pol = (rec.polarization == null || rec.polarization.isEmpty()) ? "OTHER" : rec.polarization;
            String bShape = (rec.blackShape == null || rec.blackShape.isEmpty()) ? "none" : rec.blackShape;
            String wShape = (rec.whiteShape == null || rec.whiteShape.isEmpty()) ? "none" : rec.whiteShape;

            xy.computeIfAbsent(season, s -> new LinkedHashMap<>())
                    .computeIfAbsent(pol, s -> new LinkedHashMap<>())
                    .computeIfAbsent(bShape, s -> new LinkedHashMap<>())
                    .computeIfAbsent(wShape, s -> new int[3]);

            int[] c = xy.get(season).get(pol).get(bShape).get(wShape);
            c[0]++;
            if (rec.flooding) c[1]++;
        }

        for (Map.Entry<String, Map<String, Map<String, Map<String, int[]>>>> eSeason : xy.entrySet()) {
            String season = eSeason.getKey();
            for (Map.Entry<String, Map<String, Map<String, int[]>>> ePol : eSeason.getValue().entrySet()) {
                String pol = ePol.getKey();
                for (Map.Entry<String, Map<String, int[]>> eBlack : ePol.getValue().entrySet()) {
                    String bShape = eBlack.getKey();
                    for (Map.Entry<String, int[]> eWhite : eBlack.getValue().entrySet()) {
                        String wShape = eWhite.getKey();
                        int[] c = eWhite.getValue();
                        int count = c[0];
                        int floodTrue = c[1];
                        double trueRate = (count > 0) ? (double) floodTrue / (double) count : 0.0;

                        rows.add(Arrays.asList(
                                season,
                                pol,
                                bShape,
                                wShape,
                                Integer.toString(count),
                                Integer.toString(floodTrue),
                                Double.toString(trueRate)
                        ));
                    }
                }
            }
        }

        rows.add(Collections.emptyList());

        rows.add(Collections.singletonList("DECISION_RULE"));
        rows.add(Arrays.asList(
                "Interpretation",
                "Score = w_raw_mean * (raw_mean - overall_mean)/overall_std + w_black_diam * black_diameter + w_white_diam * white_diameter + ...; Probability = 1 / (1 + exp(-Score))"
        ));

        return rows;
    }

    private static List<List<String>> buildSkippedRows(List<SkipRecord> skips) {
        List<List<String>> rows = new ArrayList<>();
        rows.add(Arrays.asList("image_name", "folder_name", "reason"));
        for (SkipRecord s : skips) {
            rows.add(Arrays.asList(s.imageName, s.folderName, s.reason));
        }
        return rows;
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

    private static void writeAutoProbabilities(Path rootFolder, List<ImageRecord> records) throws IOException {
        if (records.isEmpty()) {
            System.out.println("[WARN] Skipping Auto_Probabilities.csv because there are no image records.");
            return;
        }

        double rawSum = 0.0;
        double rawSqSum = 0.0;
        double maxBlack = 0.0;
        double maxWhite = 0.0;

        for (ImageRecord rec : records) {
            rawSum += rec.rawMean;
            rawSqSum += rec.rawMean * rec.rawMean;
            if (rec.blackDiameter > maxBlack) maxBlack = rec.blackDiameter;
            if (rec.whiteDiameter > maxWhite) maxWhite = rec.whiteDiameter;
        }

        double count = records.size();
        double mean = rawSum / count;
        double variance = Math.max(0.0, (rawSqSum / count) - (mean * mean));
        double stddev = (variance > 0.0) ? Math.sqrt(variance) : 0.0;

        if (maxBlack == 0.0) maxBlack = 1.0;
        if (maxWhite == 0.0) maxWhite = 1.0;

        Path autoPath = rootFolder.resolve("Auto_Probabilities.csv");

        try (BufferedWriter writer = Files.newBufferedWriter(autoPath, StandardCharsets.UTF_8)) {
            writer.write("image_name,folder_name,polarization,season,label_flooding,score,probability");
            writer.newLine();

            for (ImageRecord rec : records) {
                double zRaw = (stddev > 0.0) ? (rec.rawMean - mean) / stddev : 0.0;
                double normBlack = rec.blackDiameter / maxBlack;
                double normWhite = rec.whiteDiameter / maxWhite;

                double seasonBias = seasonBias(rec.season);
                double polBias = polarizationBias(rec.polarization);
                double shapeBias = shapeBias(rec.dominantShape);

                double score = zRaw + (0.6 * normBlack) + (0.6 * normWhite) + seasonBias + polBias + shapeBias;
                double probability = 1.0 / (1.0 + Math.exp(-score));

                List<String> row = Arrays.asList(
                        rec.imageName,
                        rec.folderName,
                        rec.polarization,
                        rec.season,
                        Boolean.toString(rec.flooding),
                        Double.toString(score),
                        Double.toString(probability)
                );
                writer.write(escapeCsvRow(row));
                writer.newLine();
            }
        }

        System.out.println("[INFO] Auto_Probabilities.csv written with lightweight logistic-style scores in: " + rootFolder);
    }

    private static double seasonBias(String season) {
        if (season == null) return 0.0;
        switch (season) {
            case "Summer":
                return 0.25;
            case "Spring":
            case "Fall":
                return 0.15;
            case "Winter":
                return -0.05;
            default:
                return 0.0;
        }
    }

    private static double polarizationBias(String pol) {
        if (pol == null) return 0.0;
        switch (pol.toUpperCase(Locale.ROOT)) {
            case "VH":
                return 0.2;
            case "VV":
                return 0.05;
            default:
                return 0.0;
        }
    }

    private static double shapeBias(String dominantShape) {
        if (dominantShape == null || dominantShape.isEmpty()) return 0.0;
        String lower = dominantShape.toLowerCase(Locale.ROOT);
        if (lower.contains("elongated")) return 0.1;
        if (lower.contains("compact")) return 0.05;
        if (lower.contains("spread")) return 0.02;
        return 0.0;
    }
}

