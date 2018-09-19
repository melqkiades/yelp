package org.insightcentre.richcontext;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by fpena on 09/04/2017.
 */
public class LibfmExporter extends ReviewsExporter {


    public static void main(String[] args) throws IOException {


//        List<Review> reviews = new ArrayList<>();
//
//        Review review1 = new Review(101, 101, 5);
//        Map<String, Double> contextTopics1 = new HashMap<>();
//        contextTopics1.put("business", 0.7);
//        contextTopics1.put("solo", 0.6);
//        review1.setContextTopics(contextTopics1);
//
//        Review review2 = new Review(101, 202, 3);
//        Map<String, Double> contextTopics2 = new HashMap<>();
//        contextTopics2.put("business", 0.3);
//        contextTopics2.put("family", 0.8);
//        contextTopics2.put("holiday", 0.2);
//        review2.setContextTopics(contextTopics2);
//
//        Review review3 = new Review(102, 203, 1);
//        Review review4 = new Review(103, 101, 4);
//
//        reviews.add(review1);
//        reviews.add(review2);
//        reviews.add(review3);
//        reviews.add(review4);
//
//        String filePath = "/Users/fpena/tmp/reviews.libfm";
//        Map<String, Integer> oneHotIdMap = getOneHot(reviews);
//        exportRecommendationsToCsv(reviews, filePath, oneHotIdMap);

        List<Review> trainReviews = new ArrayList<>();

        {
            Review review = new Review(1, 101, 4.0);
            Map<String, Double> context = new HashMap<>();
            context.put("business", 0.7);
            context.put("solo", 0.6);
            review.setContext(context);
            trainReviews.add(review);
        }

        {
            Review review = new Review(1, 102, 1.0);
            Map<String, Double> context = new HashMap<>();
            context.put("business", 0.7);
            context.put("solo", 0.6);
            review.setContext(context);
            trainReviews.add(review);
        }

        {
            Review review = new Review(1, 103, 3.0);
            Map<String, Double> context = new HashMap<>();
            context.put("business", 0.7);
            context.put("solo", 0.6);
            review.setContext(context);
            trainReviews.add(review);
        }

        {
            Review review = new Review(2, 101, 5.0);
            Map<String, Double> context = new HashMap<>();
            context.put("business", 0.7);
            context.put("solo", 0.6);
            review.setContext(context);
            trainReviews.add(review);
        }

        {
            Review review = new Review(2, 104, 2.0);
            Map<String, Double> context = new HashMap<>();
            context.put("business", 0.7);
            context.put("solo", 0.6);
            review.setContext(context);
            trainReviews.add(review);
        }

        List<Review> testReviews = new ArrayList<>();

        {
            Review review = new Review(1, 106, 5.0);
            Map<String, Double> context = new HashMap<>();
            context.put("business", 0.7);
            context.put("solo", 0.6);
            review.setContext(context);
            testReviews.add(review);
        }

        {
            Review review = new Review(1, 105, 1.0);
            Map<String, Double> context = new HashMap<>();
            context.put("holidays", 0.7);
//            context.put("solo", 0.6);
            review.setContext(context);
            testReviews.add(review);
        }

        {
            Review review = new Review(2, 107, 5.0);
            Map<String, Double> context = new HashMap<>();
            context.put("business", 0.7);
            context.put("solo", 0.6);
            review.setContext(context);
            testReviews.add(review);
        }

        {
            Review review = new Review(2, 105, 5.0);
            Map<String, Double> context = new HashMap<>();
            context.put("business", 0.7);
            context.put("solo", 0.6);
            review.setContext(context);
            testReviews.add(review);
        }

        List<Review> allReviews = new ArrayList<>();
        allReviews.addAll(trainReviews);
        allReviews.addAll(testReviews);
        Map<String, Integer> oneHotIdMap = getOneHot(allReviews);

        String filePath = "/Users/fpena/tmp/reviews.libfm";
        String predictionsFile = "/Users/fpena/tmp/predictions.csv";
        LibfmExporter exporter =  new LibfmExporter();
        exporter.exportRankingPredictionsFile(
                trainReviews, testReviews, filePath, oneHotIdMap, predictionsFile);
    }


@Override
    protected String getDelimiter() {
        return " ";
    }

    @Override
    protected void writeHeader(List<String> contextKeys, BufferedWriter writer) {

    }

        protected void writeLibfmReviewToFile(
            Review review, Map<String, Integer> oneHotIdMap,
            BufferedWriter bufferedWriter) throws IOException {

        String userId = "user_" + String.valueOf(review.getUserId());
        String itemId = "item_" + String.valueOf(review.getItemId());

        Map<String, Double> contextTopics = review.getContext();

        List<String> row = new ArrayList<>();
        row.add(String.valueOf(review.getRating()));
        row.add(String.valueOf(oneHotIdMap.get(userId)) + ":1");
        row.add(String.valueOf(oneHotIdMap.get(itemId)) + ":1");

        if (contextTopics != null) {
            for (String contextColumn : contextTopics.keySet()) {

                String contextId = "context_" + contextColumn;
                row.add(String.valueOf(
                        oneHotIdMap.get(contextId)) + ":" + contextTopics.get(contextColumn));
            }
        }

        bufferedWriter.write(String.join(DELIMITER, row) + "\n");
    }
}
