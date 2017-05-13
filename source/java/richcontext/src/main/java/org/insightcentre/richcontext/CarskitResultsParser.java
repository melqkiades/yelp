package org.insightcentre.richcontext;

import com.opencsv.CSVReader;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by fpena on 29/03/2017.
 */
public class CarskitResultsParser {


    private static final int USER_ID_INDEX = 0;
    private static final int ITEM_ID_INDEX = 1;
    private static final int CONTEXT_INDEX = 2;
    private static final int RATING_INDEX = 3;
    private static final int PREDICTION_INDEX = 4;

    private static final String separator = "\t";


    public static List<Review> parseRatingResults(String filePath)
            throws IOException {

        CSVReader reader = new CSVReader(new FileReader(filePath), '\t');
        List<Review> reviews = new ArrayList<>();

        String [] nextLine;

        // We ignore the header
        reader.readNext();

        while ((nextLine = reader.readNext()) != null) {
            long user_id = Long.parseLong(nextLine[USER_ID_INDEX]);
            long item_id = Long.parseLong(nextLine[ITEM_ID_INDEX]);
            double rating = Double.parseDouble(nextLine[RATING_INDEX]);
            double predictedRating =
                    Double.parseDouble(nextLine[PREDICTION_INDEX]);
            Review review = new Review(user_id, item_id, rating);
            review.setPredictedRating(predictedRating);
            reviews.add(review);
        }

        return reviews;
    }


    public static List<Review> parseRankingResults(String filePath)
            throws IOException {

        List<Review> reviews = new ArrayList<>();

        BufferedReader bufferedReader = new BufferedReader(new FileReader(filePath));
        String line;

        // Ignore the first line
        bufferedReader.readLine();

        int progress = 1;
        int totalLines = countLines(filePath);

        while ((line = bufferedReader.readLine()) != null) {
//            System.out.println(line);
            String[] userSplit = line.split(", ", 2);
            long userId = Long.parseLong(userSplit[0]);
            String[] contextSplit = userSplit[1].split(" ", 2);
            String context = contextSplit[0];
            String[] predictionsSplit = contextSplit[1].split("\\), ");
            System.out.print("Progress: " + progress + "/" + totalLines + "\r");
            progress++;
//
//
            for (String predictionString : predictionsSplit) {
////                System.out.println(predictionString);
                String[] itemPrediction =
                        predictionString.replaceAll("\\(", "").split(", ");
                long itemId = Long.parseLong(itemPrediction[0].replaceAll("\\*", ""));
                double predictedRating = Double.parseDouble(
                        itemPrediction[1].replaceAll("\\)", ""));
//
                Review review = new Review(userId, itemId);
                review.setPredictedRating(predictedRating);
                reviews.add(review);
            }
        }

        return reviews;
//        return null;


        // From the second line on

        // split the string and take the first part before ", ", leave the
        // rest as one string

        // split the remaining string and take the first part before " ", leave
        // the rest as one string

        // split the remaining string by ")," and for each token remove "("

        // For each token split by ", " first part is the itemId, second part
        // is the
    }


    public static int countLines(String filename) throws IOException {
        InputStream is = new BufferedInputStream(new FileInputStream(filename));
        try {
            byte[] c = new byte[1024];
            int count = 0;
            int readChars = 0;
            boolean empty = true;
            while ((readChars = is.read(c)) != -1) {
                empty = false;
                for (int i = 0; i < readChars; ++i) {
                    if (c[i] == '\n') {
                        ++count;
                    }
                }
            }
            return (count == 0 && !empty) ? 1 : count;
        } finally {
            is.close();
        }
    }


    public static void main(String[] args) {


        String folder = "/Users/fpena/tmp/CARSKit/context-aware_data_sets/Movie_DePaulMovie/CARSKit.Workspace/";
        String ratingsFilePath = folder + "GlobalAvg-rating-predictions fold [1].txt";
        String rankingsFilePath = folder + "PMF-top-10-items fold [1].txt";

        try {
//            List<Review> ratingReviews = parseRatingResults(ratingsFilePath);
            List<Review> rankingReviews = parseRankingResults(rankingsFilePath);

            for (Review review : rankingReviews) {
                System.out.println(review);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
