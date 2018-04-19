package org.insightcentre.richcontext;

import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarStyle;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Created by fpena on 09/04/2017.
 */
public class LibfmResultsParser {


    private static final int USER_ID_INDEX = 0;
    private static final int ITEM_ID_INDEX = 1;
    private static final int RATING_INDEX = 2;


    public static void parseResults(
            String testFilePath, String libfmResultsPath, String ouputFile)
            throws IOException {

        parseResults(testFilePath, libfmResultsPath, false, ouputFile);
    }

    public static void parseResults(
            String testFilePath, String libfmResultsPath, boolean ignoreHeader,
            String outputFile) throws IOException {

        BufferedReader testFileReader = Files.newBufferedReader(
                Paths.get(testFilePath), StandardCharsets.UTF_8);
        BufferedReader libfmResultsReader = Files.newBufferedReader(
                Paths.get(libfmResultsPath), StandardCharsets.UTF_8);
        BufferedWriter bufferedWriter =
                new BufferedWriter(new FileWriter(outputFile));

        String testLine;
        String libfmLine;

        // We ignore the header
        if (ignoreHeader) {
            testFileReader.readLine();
        }

        long numLines = Files.lines(Paths.get(testFilePath)).count();
        ProgressBar progressBar = new ProgressBar(
                "Parse rating results", numLines, 1000, System.out,
                ProgressBarStyle.ASCII);
        progressBar.start();

        while ((testLine = testFileReader.readLine()) != null) {
            libfmLine = libfmResultsReader.readLine();
            String[] lineTokens = testLine.split("\t");
            long user_id = Long.parseLong(lineTokens[USER_ID_INDEX]);
            long item_id = Long.parseLong(lineTokens[ITEM_ID_INDEX]);
            double rating = Double.parseDouble(lineTokens[RATING_INDEX]);
            double predictedRating =
                    Double.parseDouble(libfmLine);
            Review review = new Review(user_id, item_id, rating);
            review.setPredictedRating(predictedRating);
            LibfmExporter.writeReviewToFile(review, bufferedWriter);
            progressBar.step();
        }
        bufferedWriter.flush();
        bufferedWriter.close();
        progressBar.stop();
        System.out.println("\n");
    }
}
