package org.insightcentre.richcontext;

import com.opencsv.CSVWriter;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by fpena on 27/03/2017.
 */
public class CarskitExporter {

    public static void  exportWithContext(
            List<Review> reviews, String filePath) throws IOException {


        BufferedWriter bufferedWriter =
                new BufferedWriter(new FileWriter(filePath));
        String delimiter = ",";

        List<String> headers = new ArrayList<>();
        headers.add("user_id");
        headers.add("business_id");
        headers.add("stars");

        List<String> contextKeys = new ArrayList<>(reviews.get(0).getContext().keySet());
        contextKeys.remove("na");

        for (String contextKey: contextKeys) {
            headers.add("context:" + contextKey);
        }
        headers.add("context:na");

        bufferedWriter.write(String.join(delimiter, headers) + "\n");

        int totalReviews = reviews.size();
        int progress = 1;

        for (Review review : reviews) {

            System.out.print("Progress: " + progress + "/" + totalReviews + "\r");
            progress++;

//            String[] row = {
//                    String.valueOf(review.getUserId()),
//                    String.valueOf(review.getItemId()),
//                    Double.toString(review.getPredictedRating())
//            };
            List<String> row = new ArrayList<>();
            row.add(String.valueOf(review.getUserId()));
            row.add(String.valueOf(review.getItemId()));
            row.add(String.valueOf(review.getRating()));

            for (String contextKey: contextKeys) {
                row.add(String.valueOf(review.getContext().get(contextKey).intValue()));
            }
            row.add(String.valueOf(review.getContext().get("na").intValue()));

            bufferedWriter.write(String.join(delimiter, row) + "\n");
        }
        bufferedWriter.flush();
        bufferedWriter.close();
    }



    public static void exportRecommendationsToCsv(
            List<Review> reviews, String filePath) throws IOException {


        BufferedWriter bufferedWriter =
                new BufferedWriter(new FileWriter(filePath));
        String delimiter = "\t";


        int totalReviews = reviews.size();
        int progress = 1;

        for (Review review : reviews) {

//            System.out.print("Progress: " + progress + "/" + totalReviews + "\r");
            progress++;

            String[] row = {
                    String.valueOf(review.getUserId()),
                    String.valueOf(review.getItemId()),
                    Double.toString(review.getPredictedRating())
            };

            bufferedWriter.write(String.join(delimiter, row) + "\n");
        }
        bufferedWriter.flush();
        bufferedWriter.close();
    }


    public static void main(String[] args) throws IOException {

        String cacheFolder = "/Users/fpena/UCC/Thesis/datasets/context/stuff/cache_context/";
        String jsonFile = "yelp_hotel_recsys_formatted_context_records_ensemble_" +
                "numtopics-10_iterations-100_passes-10_targetreview-specific_" +
                "normalized_contextformat-predefined_context_lang-en_bow-NN_" +
                "document_level-review_targettype-context_" +
                "min_item_reviews-10.json";

        List<Review> reviews = JsonParser.readReviews(new File(cacheFolder + jsonFile));

        String outFile = "/Users/fpena/tmp/carskit_test_reviews.csv";
        exportWithContext(reviews, outFile);
    }
}
