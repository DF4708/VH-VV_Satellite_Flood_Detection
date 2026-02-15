import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.DataBuffer;

import java.io.BufferedWriter;
import java.io.IOException;

import java.nio.charset.StandardCharsets;
import java.nio.file.*;

import java.util.*;
import java.util.concurrent.*;

/**
 * Data_Extraction_M1_Optimized - **ALL RIGHTS RESERVED**
 * Expressed written and verbal permission must be given by the author for the use, modification, and or distribution of this code.
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
 *     with a fixed thread pool (optimized for Apple M1 Max: 8 threads).
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
 *          One CSV with multiple "sections" (one header):
 *            - STATS   : group stats on raw_mean (pre-mean and post-mean, std, median,
 *                        and post-mode based on RAW pixel counts, not raw_mean values).
 *                        Mode ignores RAW values whose aggregate pixel count is 0 and RAW_00000.
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
 *   - After CSV generation, automatically calls SummaryGeneratorLogistic.main(rootFolderPath)
 *     as a follow-on step.
 */
public class Data_Extraction_M1_Optimized {

    // ---------- Helper types ----------

    private static class ComponentStats {
        int size;
        int width;
        int height;
        double diameter;
        String shape;
    }

    private static class ImageAnalysis {
        Map<String, Integer> rawCounts;
        double rawMean;
        ComponentStats blackStats;
        ComponentStats whiteStats;
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
        String polarization;
        boolean flooding;
        String season;
        double rawMean;

        ComponentStats blackStats;
        ComponentStats whiteStats;
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
        final ImageRecord record;
        final SkipRecord skip;
        JobResult(ImageRecord record, SkipRecord skip) {
            this.record = record;
            this.skip = skip;
        }
    }

    private static class WeightContext {
        double overallTrueRate;
        double wRaw, meanRawAll, stdRawAll;
        double wBd, meanBdAll, stdBdAll;
        double wWd, meanWdAll, stdWdAll;
        Map<String, Double> seasonWeight = new HashMap<>();
        Map<String, Double> polWeight = new HashMap<>();
        Map<String, Double> blackShapeWeight = new HashMap<>();
        Map<String, Double> whiteShapeWeight = new HashMap<>();
    }

    // ---------- MAIN ----------

    public static void main(String[] args) {
        Path rootFolder = (args.length > 0)
                ? Paths.get(args[0]).toAbsolutePath()
                : Paths.get("").toAbsolutePath();

        System.out.println("[INFO] Root folder: " + rootFolder);

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
                    return processSingleImage(job, floodBySentinelName);
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
        System.out.println(); // end progress line

        System.out.println("[INFO] Images successfully processed: " + records.size());
        System.out.println("[INFO] Images skipped: " + skips.size());

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

        // ---------- Follow-on script: SummaryGeneratorLogistic ----------
        System.out.println("[INFO] Launching SummaryGeneratorLogistic...");
        try {
            // Pass the root folder path so the follow-on script knows where the CSVs are.
            SummaryGeneratorLogistic.main(new String[]{ rootFolder.toString() });
            System.out.println("[INFO] SummaryGeneratorLogistic completed.");
        } catch (Throwable t) {
            System.err.println("[WARN] Failed to run SummaryGeneratorLogistic: " + t.getMessage());
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
            System.err.println("[WARN] Could not list root directory: " + e.getMessage());
        }

        // Root folder itself (folderName = "ROOT").
        count += gatherJobsInFolder(rootFolder, "ROOT", jobs);

        return count;
    }

    private static int gatherJobsInFolder(Path folderPath, String folderName, List<ImageJob> jobs) {
        int count = 0;
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(folderPath)) {
            for (Path entry : ds) {
                if (Files.isDirectory(entry)) continue;
                String fileName = entry.getFileName().toString();
                String lower = fileName.toLowerCase();
                if (lower.endsWith(".tif") || lower.endsWith(".tiff")) {
                    jobs.add(new ImageJob(entry, folderName));
                    count++;
                }
            }
        } catch (IOException e) {
            System.err.println("[WARN] Could not list folder " + folderPath + ": " + e.getMessage());
        }
        return count;
    }

    // ---------- Per-image processing ----------

    private static JobResult processSingleImage(
            ImageJob job,
            Map<String, Boolean> floodBySentinelName
    ) {
        Path imagePath = job.imagePath;
        String folderName = job.folderName;
        String fileName = imagePath.getFileName().toString();

        String baseName = stripExtension(fileName);
        Boolean floodFlag = findFloodFlagForImage(baseName, floodBySentinelName);

        if (floodFlag == null) {
            return new JobResult(
                    null,
                    new SkipRecord(fileName, folderName, "No matching FLOODING label in S1list/S2list")
            );
        }

        BufferedImage image;
        try {
            image = javax.imageio.ImageIO.read(imagePath.toFile());
        } catch (IOException e) {
            return new JobResult(
                    null,
                    new SkipRecord(fileName, folderName,
                            "Failed to read image (TIFF decoding error). Consider external pre-conversion.")
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

        if (fileName.contains("VV")) {
            record.polarization = "VV";
        } else if (fileName.contains("VH")) {
            record.polarization = "VH";
        } else {
            record.polarization = "OTHER";
        }

        record.rawMean = analysis.rawMean;
        record.blackStats = analysis.blackStats;
        record.whiteStats = analysis.whiteStats;
        record.dominantShape = analysis.dominantShape;
        record.rawCounts = analysis.rawCounts;

        return new JobResult(record, null);
    }

    // ---------- JSON label parsing ----------

    private static String stripExtension(String fileName) {
        int dot = fileName.lastIndexOf('.');
        if (dot < 0) return fileName;
        return fileName.substring(0, dot);
    }

    private static Boolean findFloodFlagForImage(
            String imageBaseName,
            Map<String, Boolean> floodBySentinelName
    ) {
        for (Map.Entry<String, Boolean> entry : floodBySentinelName.entrySet()) {
            String sentinelName = entry.getKey();
            if (imageBaseName.contains(sentinelName)) {
                return entry.getValue();
            }
        }
        return null;
    }

    private static void parseFloodJsonFile(
            Path jsonPath,
            Map<String, Boolean> floodBySentinelName
    ) throws IOException {
        String json = Files.readString(jsonPath, StandardCharsets.UTF_8);

        int searchIndex = 0;
        while (true) {
            int filenameKeyIndex = json.indexOf("\"filename\"", searchIndex);
            if (filenameKeyIndex < 0) break;

            int colonIndex = json.indexOf(':', filenameKeyIndex);
            if (colonIndex < 0) break;

            int firstQuote = json.indexOf('"', colonIndex + 1);
            int secondQuote = json.indexOf('"', firstQuote + 1);
            if (firstQuote < 0 || secondQuote < 0) break;

            String fullFilename = json.substring(firstQuote + 1, secondQuote).trim();
            String baseName = fullFilename;
            int dot = baseName.lastIndexOf('.');
            if (dot >= 0) baseName = baseName.substring(0, dot);

            int windowStart = Math.max(0, filenameKeyIndex - 300);
            int floodingIndex = json.lastIndexOf("\"FLOODING\"", filenameKeyIndex);
            if (floodingIndex < windowStart) floodingIndex = -1;

            boolean flooding = false;
            if (floodingIndex >= 0) {
                int colonFlood = json.indexOf(':', floodingIndex);
                if (colonFlood > 0) {
                    int trueIndex = json.indexOf("true", colonFlood);
                    int falseIndex = json.indexOf("false", colonFlood);
                    if (trueIndex > 0 && trueIndex < secondQuote &&
                            (falseIndex < 0 || trueIndex < falseIndex)) {
                        flooding = true;
                    } else if (falseIndex > 0 && falseIndex < secondQuote) {
                        flooding = false;
                    }
                }
            }

            floodBySentinelName.put(baseName, flooding);
            searchIndex = secondQuote + 1;
        }

        System.out.println("[INFO] Parsed flood labels from " + jsonPath.getFileName());
    }

    // ---------- Per-image analysis: RAW histogram + shapes (non-overlap + two-pass + fallback) ----------

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
                rawCounts.put(key, rawCounts.getOrDefault(key, 0) + 1);

                if (darkSet.contains(raw))  blackRow[x] = true;
                if (brightSet.contains(raw)) whiteRow[x] = true;
            }
        }

        // First pick white component; enforce area/density thresholds (non-overlap logic).
        ComponentStats whiteStats = largestComponentStats(
                whiteMask, width, height, true);

        // Remove white area from black mask to ensure no overlap.
        if (whiteStats != null && whiteStats.size > 0) {
            for (int y = 0; y < height; y++) {
                boolean[] bRow = blackMask[y];
                boolean[] wRow = whiteMask[y];
                for (int x = 0; x < width; x++) {
                    if (wRow[x]) bRow[x] = false;
                }
            }
        }

        ComponentStats blackStats = largestComponentStats(
                blackMask, width, height, false);

        if (whiteStats == null) whiteStats = emptyStats();
        if (blackStats == null) blackStats = emptyStats();

        String dominantShape;
        if (blackStats.size >= whiteStats.size) {
            dominantShape = blackStats.shape;
        } else {
            dominantShape = whiteStats.shape;
        }

        ImageAnalysis result = new ImageAnalysis();
        result.rawCounts = rawCounts;
        result.rawMean = rawMean;
        result.blackStats = blackStats;
        result.whiteStats = whiteStats;
        result.dominantShape = dominantShape;
        return result;
    }

    private static ComponentStats emptyStats() {
        ComponentStats cs = new ComponentStats();
        cs.size = 0;
        cs.width = 0;
        cs.height = 0;
        cs.diameter = 0.0;
        cs.shape = "none";
        return cs;
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

    // ---------- Connected components with two-pass thresholds + fallback ----------

    private static ComponentStats largestComponentStats(
            boolean[][] mask,
            int imageWidth,
            int imageHeight,
            boolean isWhite
    ) {
        // Pass 1: strict (avoid huge, low-density blobs)
        ComponentStats strict = largestComponentStatsWithThresholds(
                mask,
                imageWidth,
                imageHeight,
                0.40,   // max 40% of full image area
                0.10,   // at least 10% of bounding box filled
                20      // min 20 pixels
        );
        if (strict != null) {
            return strict;
        }

        // Pass 2: relaxed (allow more shapes but still avoid almost-full-image blobs)
        ComponentStats relaxed = largestComponentStatsWithThresholds(
                mask,
                imageWidth,
                imageHeight,
                0.95,   // max 95% of full image area
                0.001,  // at least 0.1% of bounding box filled (more tolerant)
                20      // min 20 pixels
        );
        if (relaxed != null) {
            return relaxed;
        }

        // Fallback: pick largest connected component with only a minimum-size threshold.
        ComponentStats fallback = largestComponentStatsNoFilter(mask, imageWidth, imageHeight, 20);
        if (fallback != null) {
            return fallback;
        }

        // If nothing at all is found, return empty stats.
        return emptyStats();
    }

    private static ComponentStats largestComponentStatsWithThresholds(
            boolean[][] mask,
            int imageWidth,
            int imageHeight,
            double maxAreaRatio,
            double minFillRatio,
            int minSize
    ) {
        int height = mask.length;
        if (height == 0) return null;
        int width = mask[0].length;

        boolean[][] visited = new boolean[height][width];

        ComponentStats best = null;
        int[] dX = {1, -1, 0, 0};
        int[] dY = {0, 0, 1, -1};

        double imageArea = (double) imageWidth * (double) imageHeight;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (!mask[y][x] || visited[y][x]) continue;

                int minX = x, maxX = x;
                int minY = y, maxY = y;
                int size = 0;

                Deque<int[]> queue = new ArrayDeque<>();
                queue.add(new int[]{x, y});
                visited[y][x] = true;

                while (!queue.isEmpty()) {
                    int[] p = queue.removeFirst();
                    int px = p[0], py = p[1];
                    size++;

                    if (px < minX) minX = px;
                    if (px > maxX) maxX = px;
                    if (py < minY) minY = py;
                    if (py > maxY) maxY = py;

                    for (int k = 0; k < 4; k++) {
                        int nx = px + dX[k];
                        int ny = py + dY[k];
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                        if (!mask[ny][nx] || visited[ny][nx]) continue;
                        visited[ny][nx] = true;
                        queue.addLast(new int[]{nx, ny});
                    }
                }

                if (size < minSize) continue; // too small to be meaningful

                int compWidth = maxX - minX + 1;
                int compHeight = maxY - minY + 1;
                double boxArea = (double) compWidth * (double) compHeight;

                double areaRatio = boxArea / imageArea;
                double fillRatio = (boxArea > 0.0) ? ((double) size / boxArea) : 0.0;

                if (areaRatio > maxAreaRatio) continue;    // too big (covers too much of image)
                if (fillRatio < minFillRatio) continue;    // too sparse / outline-like

                double diameter = Math.sqrt(
                        (double) compWidth * compWidth +
                                (double) compHeight * compHeight
                );

                if (best == null || size > best.size) {
                    ComponentStats cs = new ComponentStats();
                    cs.size = size;
                    cs.width = compWidth;
                    cs.height = compHeight;
                    cs.diameter = diameter;
                    cs.shape = classifyShape(size, compWidth, compHeight);
                    best = cs;
                }
            }
        }

        return best;
    }

    /**
     * Fallback connected-component search:
     * no area/fill ratio thresholds, only a minimum size.
     * Ensures that if any contiguous region exists, we return it
     * instead of reporting "none".
     */
    private static ComponentStats largestComponentStatsNoFilter(
            boolean[][] mask,
            int imageWidth,
            int imageHeight,
            int minSize
    ) {
        int height = mask.length;
        if (height == 0) return null;
        int width = mask[0].length;

        boolean[][] visited = new boolean[height][width];

        ComponentStats best = null;
        int[] dX = {1, -1, 0, 0};
        int[] dY = {0, 0, 1, -1};

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (!mask[y][x] || visited[y][x]) continue;

                int minX = x, maxX = x;
                int minY = y, maxY = y;
                int size = 0;

                Deque<int[]> queue = new ArrayDeque<>();
                queue.add(new int[]{x, y});
                visited[y][x] = true;

                while (!queue.isEmpty()) {
                    int[] p = queue.removeFirst();
                    int px = p[0], py = p[1];
                    size++;

                    if (px < minX) minX = px;
                    if (px > maxX) maxX = px;
                    if (py < minY) minY = py;
                    if (py > maxY) maxY = py;

                    for (int k = 0; k < 4; k++) {
                        int nx = px + dX[k];
                        int ny = py + dY[k];
                        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                        if (!mask[ny][nx] || visited[ny][nx]) continue;
                        visited[ny][nx] = true;
                        queue.addLast(new int[]{nx, ny});
                    }
                }

                if (size < minSize) continue;

                int compWidth = maxX - minX + 1;
                int compHeight = maxY - minY + 1;

                double diameter = Math.sqrt(
                        (double) compWidth * compWidth +
                                (double) compHeight * compHeight
                );

                if (best == null || size > best.size) {
                    ComponentStats cs = new ComponentStats();
                    cs.size = size;
                    cs.width = compWidth;
                    cs.height = compHeight;
                    cs.diameter = diameter;
                    cs.shape = classifyShape(size, compWidth, compHeight);
                    best = cs;
                }
            }
        }

        return best;
    }

    /**
     * Shape heuristic:
     * - Circle only for very compact, nearly square components.
     * - Others: square, ellipse, rectangle, parallelogram, trapezium, triangle, crescent.
     */
    private static String classifyShape(int size, int width, int height) {
        if (size <= 0 || width <= 0 || height <= 0) return "none";

        int areaBox = width * height;
        double fillRatio = (areaBox > 0) ? ((double) size / (double) areaBox) : 0.0;
        double aspect = (width >= height)
                ? (double) width / (double) height
                : (double) height / (double) width;

        if (fillRatio >= 0.80 && aspect <= 1.05) {
            return "circle";
        }
        if (fillRatio >= 0.65 && aspect <= 1.15) {
            return "square";
        }
        if (fillRatio >= 0.55 && aspect <= 1.6) {
            return "ellipse";
        }
        if (fillRatio >= 0.50 && aspect > 1.6 && aspect <= 3.0) {
            return "rectangle";
        }
        if (fillRatio >= 0.35) {
            if (aspect <= 1.4) {
                return "parallelogram";
            } else {
                return "trapezium";
            }
        }
        if (aspect > 1.6) {
            return "crescent";
        }
        return "triangle";
    }

    // ---------- Season inference ----------

    private static String inferSeasonFromFilename(String imageName) {
        if (imageName == null) return null;
        String digits = imageName.replaceAll("[^0-9]", " ");
        String[] parts = digits.trim().split("\\s+");
        for (String p : parts) {
            if (p.length() == 8) {
                try {
                    int year = Integer.parseInt(p.substring(0, 4));
                    int month = Integer.parseInt(p.substring(4, 6));
                    int day = Integer.parseInt(p.substring(6, 8));
                    if (year < 1900 || month < 1 || month > 12 || day < 1 || day > 31) continue;
                    return seasonFromMonth(month);
                } catch (NumberFormatException e) {
                    // ignore
                }
            }
        }
        return "Unknown";
    }

    private static String seasonFromMonth(int month) {
        switch (month) {
            case 12:
            case 1:
            case 2:
                return "Winter";
            case 3:
            case 4:
            case 5:
                return "Spring";
            case 6:
            case 7:
            case 8:
                return "Summer";
            case 9:
            case 10:
            case 11:
                return "Autumn";
            default:
                return "Unknown";
        }
    }

    // ---------- Images_All.csv ----------

    private static List<List<String>> buildImagesAllRows(
            List<ImageRecord> records,
            Set<String> allRawCodes
    ) {
        List<List<String>> rows = new ArrayList<>();

        List<String> header = new ArrayList<>();
        header.add("image_name");
        header.add("folder_name");
        header.add("polarization");
        header.add("flooding");
        header.add("season");
        header.add("raw_mean");

        header.add("black_component_size");
        header.add("black_width");
        header.add("black_height");
        header.add("black_diameter");
        header.add("black_shape");

        header.add("white_component_size");
        header.add("white_width");
        header.add("white_height");
        header.add("white_diameter");
        header.add("white_shape");

        header.add("dominant_shape");

        for (String code : allRawCodes) {
            header.add(code);
        }

        rows.add(header);

        for (ImageRecord rec : records) {
            List<String> row = new ArrayList<>();
            row.add(rec.imageName);
            row.add(rec.folderName);
            row.add(rec.polarization);
            row.add(Boolean.toString(rec.flooding));
            row.add(rec.season == null ? "" : rec.season);
            row.add(Double.toString(rec.rawMean));

            row.add(Integer.toString(rec.blackStats.size));
            row.add(Integer.toString(rec.blackStats.width));
            row.add(Integer.toString(rec.blackStats.height));
            row.add(Double.toString(rec.blackStats.diameter));
            row.add(rec.blackStats.shape);

            row.add(Integer.toString(rec.whiteStats.size));
            row.add(Integer.toString(rec.whiteStats.width));
            row.add(Integer.toString(rec.whiteStats.height));
            row.add(Double.toString(rec.whiteStats.diameter));
            row.add(rec.whiteStats.shape);

            row.add(rec.dominantShape);

            for (String code : allRawCodes) {
                Integer count = rec.rawCounts.get(code);
                row.add(count == null ? "0" : count.toString());
            }

            rows.add(row);
        }

        return rows;
    }

    // ---------- Summary_All.csv (multi-section) ----------

    private static List<List<String>> buildSummaryAllRows(List<ImageRecord> records) {
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
        out.add(header);

        if (records.isEmpty()) {
            out.add(row7("NOTE", "no_data", "", "", "", "",
                    "No images produced data rows; summary is empty."));
            return out;
        }

        out.addAll(buildStatsSection(records));
        out.addAll(buildSeasonsSection(records));
        out.addAll(buildShapesSection(records));
        out.addAll(buildWeightsSection(records));
        out.addAll(buildXYTableSection(records));

        List<String> ruleRow = new ArrayList<>();
        ruleRow.add("DECISION_RULE");
        ruleRow.add("score_formula");
        ruleRow.add("");
        ruleRow.add("");
        ruleRow.add("");
        ruleRow.add("");
        ruleRow.add("score = w_raw*z_raw_mean + w_bd*z_black_diameter + w_wd*z_white_diameter + w_season + w_pol + w_black_shape + w_white_shape; z_feature = (x - mean_all)/std_all; P(FLOODING=true) = 1/(1+exp(-score)).");
        out.add(ruleRow);

        return out;
    }

    // ---------- STATS section (post-mode from RAW pixel counts) ----------

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

        String medianTrueStr  = valsTrueNoNoZero.isEmpty()  ? "" : Double.toString(medianTrueNo);
        String medianFalseStr = valsFalseNoNoZero.isEmpty() ? "" : Double.toString(medianFalseNo);

        // Compute mode from RAW pixel counts across post-mean images.
        Map<Integer, Long> rawCountsTrue = new HashMap<>();
        Map<Integer, Long> rawCountsFalse = new HashMap<>();

        // Precompute thresholds to decide if an image is post-mean or not,
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
                "Pre-mean raw_mean across all images (zeros ignored at pixel level)."));

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
                "Post-mean raw_mean after 3-sigma outlier removal (pixel-level zeros ignored)."));

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
                "Post-mode RAW value: RAW sample whose total pixel count is highest across post-mean images in each group; zero-count RAW values are ignored and RAW_00000 is skipped."));

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

    private static List<String> row7(String section, String name,
                                     String v1, String v2, String v3, String v4, String notes) {
        List<String> r = new ArrayList<>();
        r.add(section);
        r.add(name);
        r.add(v1 == null ? "" : v1);
        r.add(v2 == null ? "" : v2);
        r.add(v3 == null ? "" : v3);
        r.add(v4 == null ? "" : v4);
        r.add(notes == null ? "" : notes);
        return r;
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
            String bShape = rec.blackStats.shape == null ? "none" : rec.blackStats.shape;
            String wShape = rec.whiteStats.shape == null ? "none" : rec.whiteStats.shape;
            if (rec.flooding) {
                blackTrue.put(bShape, blackTrue.getOrDefault(bShape, 0) + 1);
                whiteTrue.put(wShape, whiteTrue.getOrDefault(wShape, 0) + 1);
            } else {
                blackFalse.put(bShape, blackFalse.getOrDefault(bShape, 0) + 1);
                whiteFalse.put(wShape, whiteFalse.getOrDefault(wShape, 0) + 1);
            }
        }

        out.add(row7("SHAPES", "NOTE",
                "",
                "",
                "",
                "",
                "Each image contributes one black and one white largest component (after filters and fallback)."));

        Set<String> allShapesBlack = new TreeSet<>(blackTrue.keySet());
        allShapesBlack.addAll(blackFalse.keySet());
        for (String s : allShapesBlack) {
            int ct = blackTrue.getOrDefault(s, 0);
            int cf = blackFalse.getOrDefault(s, 0);
            out.add(row7("SHAPES", "black_" + s,
                    Integer.toString(ct),
                    Integer.toString(cf),
                    "",
                    "",
                    "Largest black-component shape = " + s));
        }

        Set<String> allShapesWhite = new TreeSet<>(whiteTrue.keySet());
        allShapesWhite.addAll(whiteFalse.keySet());
        for (String s : allShapesWhite) {
            int ct = whiteTrue.getOrDefault(s, 0);
            int cf = whiteFalse.getOrDefault(s, 0);
            out.add(row7("SHAPES", "white_" + s,
                    Integer.toString(ct),
                    Integer.toString(cf),
                    "",
                    "",
                    "Largest white-component shape = " + s));
        }

        return out;
    }

    // ---------- WEIGHTS section (numeric + season + shape + polarization) ----------

    private static List<List<String>> buildWeightsSection(List<ImageRecord> records) {
        List<List<String>> out = new ArrayList<>();
        WeightContext ctx = computeWeights(records);

        out.add(row7("WEIGHTS", "raw_mean",
                Double.toString(ctx.wRaw),
                Double.toString(ctx.meanRawAll),
                Double.toString(ctx.stdRawAll),
                "",
                "Numeric weight: Cohen's d on raw_mean (per-image mean over non-zero pixels)."));

        out.add(row7("WEIGHTS", "black_diameter",
                Double.toString(ctx.wBd),
                Double.toString(ctx.meanBdAll),
                Double.toString(ctx.stdBdAll),
                "",
                "Numeric weight: Cohen's d on black diameter."));

        out.add(row7("WEIGHTS", "white_diameter",
                Double.toString(ctx.wWd),
                Double.toString(ctx.meanWdAll),
                Double.toString(ctx.stdWdAll),
                "",
                "Numeric weight: Cohen's d on white diameter."));

        for (Map.Entry<String, Double> e : ctx.seasonWeight.entrySet()) {
            String s = e.getKey();
            double w = e.getValue();
            out.add(row7("WEIGHTS", "season_" + s,
                    Double.toString(w),
                    "",
                    Double.toString(ctx.overallTrueRate),
                    "",
                    "Season weight = true_rate(season) - overall_true_rate (only shown if season sample size >= threshold)."));
        }

        for (Map.Entry<String, Double> e : ctx.polWeight.entrySet()) {
            String pol = e.getKey();
            double w = e.getValue();
            out.add(row7("WEIGHTS", "pol_" + pol,
                    Double.toString(w),
                    "",
                    Double.toString(ctx.overallTrueRate),
                    "",
                    "Polarization weight = true_rate(pol) - overall_true_rate."));
        }

        for (Map.Entry<String, Double> e : ctx.blackShapeWeight.entrySet()) {
            String s = e.getKey();
            double w = e.getValue();
            out.add(row7("WEIGHTS", "black_shape_" + s,
                    Double.toString(w),
                    "",
                    Double.toString(ctx.overallTrueRate),
                    "",
                    "Black shape weight = true_rate(shape) - overall_true_rate."));
        }

        for (Map.Entry<String, Double> e : ctx.whiteShapeWeight.entrySet()) {
            String s = e.getKey();
            double w = e.getValue();
            out.add(row7("WEIGHTS", "white_shape_" + s,
                    Double.toString(w),
                    "",
                    Double.toString(ctx.overallTrueRate),
                    "",
                    "White shape weight = true_rate(shape) - overall_true_rate."));
        }

        return out;
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
            double bd = rec.blackStats.diameter;
            double wd = rec.whiteStats.diameter;

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

        // Season weights
        Map<String, Integer> seasonTrue = new HashMap<>();
        Map<String, Integer> seasonFalse = new HashMap<>();
        for (ImageRecord rec : records) {
            String s = (rec.season == null || rec.season.isEmpty()) ? "Unknown" : rec.season;
            if (rec.flooding) seasonTrue.put(s, seasonTrue.getOrDefault(s, 0) + 1);
            else seasonFalse.put(s, seasonFalse.getOrDefault(s, 0) + 1);
        }
        String[] seasons = new String[]{"Winter", "Spring", "Summer", "Autumn", "Unknown"};
        int minSeasonCount = 30; // below this, treat weight as 0 (insufficient evidence)
        for (String s : seasons) {
            int ct = seasonTrue.getOrDefault(s, 0);
            int cf = seasonFalse.getOrDefault(s, 0) + 1; // +1 for slight smoothing
            int total = ct + cf;
            if (total == 0) continue;
            if (total < minSeasonCount) {
                ctx.seasonWeight.put(s, 0.0);
                continue;
            }
            double rate = (double) ct / (double) total;
            ctx.seasonWeight.put(s, rate - ctx.overallTrueRate);
        }

        // Polarization weights
        Map<String, Integer> polTrue = new HashMap<>();
        Map<String, Integer> polFalse = new HashMap<>();
        for (ImageRecord rec : records) {
            String p = rec.polarization == null ? "OTHER" : rec.polarization;
            if (rec.flooding) polTrue.put(p, polTrue.getOrDefault(p, 0) + 1);
            else polFalse.put(p, polFalse.getOrDefault(p, 0) + 1);
        }
        Set<String> allPol = new TreeSet<>(polTrue.keySet());
        allPol.addAll(polFalse.keySet());
        for (String p : allPol) {
            int ct = polTrue.getOrDefault(p, 0);
            int cf = polFalse.getOrDefault(p, 0) + 1; // +1 smoothing
            int total = ct + cf;
            if (total == 0) continue;
            double rate = (double) ct / (double) total;
            ctx.polWeight.put(p, rate - ctx.overallTrueRate);
        }

        // Shape weights
        Map<String, Integer> blackTrue = new HashMap<>();
        Map<String, Integer> blackFalse = new HashMap<>();
        Map<String, Integer> whiteTrue = new HashMap<>();
        Map<String, Integer> whiteFalse = new HashMap<>();
        for (ImageRecord rec : records) {
            String b = rec.blackStats.shape == null ? "none" : rec.blackStats.shape;
            String w = rec.whiteStats.shape == null ? "none" : rec.whiteStats.shape;
            if (rec.flooding) {
                blackTrue.put(b, blackTrue.getOrDefault(b, 0) + 1);
                whiteTrue.put(w, whiteTrue.getOrDefault(w, 0) + 1);
            } else {
                blackFalse.put(b, blackFalse.getOrDefault(b, 0) + 1);
                whiteFalse.put(w, whiteFalse.getOrDefault(w, 0) + 1);
            }
        }
        Set<String> allB = new TreeSet<>(blackTrue.keySet());
        allB.addAll(blackFalse.keySet());
        for (String s : allB) {
            int ct = blackTrue.getOrDefault(s, 0);
            int cf = blackFalse.getOrDefault(s, 0) + 1; // smoothing
            int total = ct + cf;
            if (total == 0) continue;
            double rate = (double) ct / (double) total;
            ctx.blackShapeWeight.put(s, rate - ctx.overallTrueRate);
        }

        Set<String> allW = new TreeSet<>(whiteTrue.keySet());
        allW.addAll(whiteFalse.keySet());
        for (String s : allW) {
            int ct = whiteTrue.getOrDefault(s, 0);
            int cf = whiteFalse.getOrDefault(s, 0) + 1; // smoothing
            int total = ct + cf;
            if (total == 0) continue;
            double rate = (double) ct / (double) total;
            ctx.whiteShapeWeight.put(s, rate - ctx.overallTrueRate);
        }

        return ctx;
    }


    // ---------- XY_TABLE section (empirical probability by season/pol/shapes within |z_raw_mean| <= 1) ----------

    private static List<List<String>> buildXYTableSection(List<ImageRecord> records) {
        List<List<String>> out = new ArrayList<>();

        // Header row describing columns:
        out.add(row7(
                "XY_TABLE", "COLUMNS",
                "total_images",
                "prob_true_percent",
                "count_true",
                "count_false",
                "For XY_TABLE rows: metric_name encodes season, polarization, black_shape, white_shape; " +
                        "value_a = total images in this combination (within |z_raw_mean| <= 1)," +
                        " value_b = P(FLOODING=true)%," +
                        " value_c = number of true-labelled images, value_d = number of false-labelled images."));

        if (records.isEmpty()) {
            return out;
        }

        // Overall mean/std of rawMean for z-score thresholding
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
            // Can't form z-scores; just bail with header only.
            return out;
        }

        double zThreshold = 1.0; // within one std dev of overall mean

        // Aggregate counts keyed by (season, pol, black_shape, white_shape)
        Map<String, int[]> comboCounts = new TreeMap<>();
        for (ImageRecord r : records) {
            double v = r.rawMean;
            if (Double.isNaN(v)) continue;
            double z = (v - mean) / std;
            if (Math.abs(z) > zThreshold) continue;

            String season = (r.season == null || r.season.isEmpty()) ? "Unknown" : r.season;
            String pol = (r.polarization == null) ? "" : r.polarization;

            String bs = (r.blackStats == null || r.blackStats.shape == null || r.blackStats.shape.isEmpty())
                    ? "none" : r.blackStats.shape;

            String ws = (r.whiteStats == null || r.whiteStats.shape == null || r.whiteStats.shape.isEmpty())
                    ? "none" : r.whiteStats.shape;

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
                    Integer.toString(total),            // value_a: total images
                    Double.toString(prob),             // value_b: probability %
                    Integer.toString(cTrue),           // value_c: true count
                    Integer.toString(cFalse),          // value_d: false count
                    ""));
        }

        return out;
    }

    // ---------- Skipped.csv ----------

    private static List<List<String>> buildSkippedRows(List<SkipRecord> skips) {
        List<List<String>> rows = new ArrayList<>();
        List<String> header = new ArrayList<>();
        header.add("image_name");
        header.add("folder_name");
        header.add("reason");
        rows.add(header);

        for (SkipRecord s : skips) {
            List<String> row = new ArrayList<>();
            row.add(s.imageName);
            row.add(s.folderName);
            row.add(s.reason);
            rows.add(row);
        }
        return rows;
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

    // Remove exact zeros from a list (used for median so that 0.0
    // does not dominate when it is just a background value).
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

    // ---------- CSV writer ----------

    private static void writeCsv(Path path, List<List<String>> rows) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            for (List<String> row : rows) {
                StringBuilder line = new StringBuilder();
                for (int i = 0; i < row.size(); i++) {
                    if (i > 0) line.append(',');
                    String field = row.get(i);
                    if (field == null) field = "";
                    String escaped = field.replace("\"", "\"\"");
                    if (escaped.contains(",") || escaped.contains("\"") ||
                            escaped.contains("\n") || escaped.contains("\r")) {
                        line.append('"').append(escaped).append('"');
                    } else {
                        line.append(escaped);
                    }
                }
                writer.write(line.toString());
                writer.newLine();
            }
        }
    }
}
