package org.insightcentre.richcontext;

import com.opencsv.CSVReader;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by fpena on 09/04/2017.
 */
public class LibfmResultsParser {


    private static final int USER_ID_INDEX = 0;
    private static final int ITEM_ID_INDEX = 1;
    private static final int RATING_INDEX = 2;


    public static List<Review> parseRatingResults(
            String testFilePath, String libfmResultsPath) throws IOException {

        BufferedReader testFileReader = Files.newBufferedReader(
                Paths.get(testFilePath), StandardCharsets.UTF_8);
        BufferedReader libfmResultsReader = Files.newBufferedReader(
                Paths.get(libfmResultsPath), StandardCharsets.UTF_8);
        List<Review> reviews = new ArrayList<>();

        String testLine;
        String libfmLine;

        // We ignore the header
//        reader.readNext();

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
            reviews.add(review);
        }

        return reviews;
    }


    public static List<Review> parseContextRatingResults(
            String testFilePath, String libfmResultsPath) throws IOException {

        BufferedReader testFileReader = Files.newBufferedReader(
                Paths.get(testFilePath), StandardCharsets.UTF_8);
        BufferedReader libfmResultsReader = Files.newBufferedReader(
                Paths.get(libfmResultsPath), StandardCharsets.UTF_8);
        List<Review> reviews = new ArrayList<>();

        String testLine;
        String libfmLine;

        // Parse the header
        testLine = testFileReader.readLine();
        String[] headers = testLine.split("\t");

        while ((testLine = testFileReader.readLine()) != null) {
            libfmLine = libfmResultsReader.readLine();
            String[] lineTokens = testLine.split("\t");
            long user_id = Long.parseLong(lineTokens[USER_ID_INDEX]);
            long item_id = Long.parseLong(lineTokens[ITEM_ID_INDEX]);
            double rating = Double.parseDouble(lineTokens[RATING_INDEX]);
            double predictedRating =
                    Double.parseDouble(libfmLine);
            Review review = new Review(user_id, item_id, rating);
            Map<String, Double> contextMap = new HashMap<>();

            for (int i = 3; i < lineTokens.length; i++) {
                contextMap.put(headers[i], Double.parseDouble(lineTokens[i]));
            }

            review.setPredictedRating(predictedRating);
            review.setContext(contextMap);
            reviews.add(review);
        }

        return reviews;
    }


    public static void main(String[] args) {


        String folder = "/Users/fpena/UCC/Thesis/datasets/context/stuff/" +
                "cache_context/rival/yelp_hotel_recsys_contextual_records_" +
                "ensemble_numtopics-30_iterations-100_passes-10_" +
                "targetreview-specific_normalized_lang-en_bow-NN_" +
                "document_level-review_targettype-context_" +
                "min_item_reviews-10/fold_0/";
        String testFilePath = folder + "test.csv";
        String libfmResultsFilePath = folder + "libfm_predictions.txt";
//        String rankingsFilePath = folder + "PMF-top-10-items fold [1].txt";

        try {
            List<Review> ratingReviews =
                    parseRatingResults(testFilePath, libfmResultsFilePath);
//            List<Review> rankingReviews = parseRankingResults(rankingsFilePath);

            for (Review review : ratingReviews) {
                System.out.println(review);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
