package org.insightcentre.richcontext;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Created by fpena on 27/03/2017.
 */
public class CarskitExporter extends ReviewsExporter {


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

        System.out.println(cacheFolder + jsonFile);

        List<Review> reviews = JsonParser.readReviews(new File(cacheFolder + jsonFile));

        String outFile = "/tmp/carskit_test_reviews.csv";
        String outFile2 = "/tmp/carskit_test_reviews2.csv";
        exportWithContext(reviews, outFile);
        CarskitExporter exporter = new CarskitExporter();
        exporter.exportRecommendations(reviews, outFile2, null);
    }

    @Override
    protected String getDelimiter() {
        return ",";
    }

    @Override
    protected void writeHeader(List<String> contextKeys, BufferedWriter writer) throws IOException {

        List<String> headers = new ArrayList<>();
        headers.add("user_id");
        headers.add("business_id");
        headers.add("stars");

        contextKeys.remove("na");

        for (String contextKey: contextKeys) {
            headers.add("context:" + contextKey);
        }
        headers.add("context:na");

        writer.write(String.join(DELIMITER, headers) + "\n");
    }

    @Override
    protected void writeLibfmReviewToFile(Review review, Map<String, Integer> oneHotIdMap, BufferedWriter bufferedWriter) throws IOException {

        List<String> row = new ArrayList<>();
        row.add(String.valueOf(review.getUserId()));
        row.add(String.valueOf(review.getItemId()));
        row.add(String.valueOf(review.getRating()));

        List<String> contextKeys = new ArrayList<>(review.getContext().keySet());
        contextKeys.remove("na");
        for (String contextColumn : contextKeys) {
            row.add(String.valueOf(review.getContext().get(contextColumn).intValue()));
        }
        row.add(String.valueOf(review.getContext().get("na").intValue()));

        bufferedWriter.write(String.join(DELIMITER, row) + "\n");

    }
}
