package org.insightcentre.richcontext;

import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarStyle;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Created by fpena on 09/04/2017.
 */
public class LibfmExporter {


    public static final double RELEVANCE_THRESHOLD = 5.0;


    public static void exportRecommendations(
            List<Review> reviews, String filePath, Map<String, Integer> oneHotIdMap) throws IOException {

        BufferedWriter bufferedWriter =
                new BufferedWriter(new FileWriter(filePath));
        String delimiter = " ";


        int totalReviews = reviews.size();
//        int progress = 1;
        ProgressBar progressBar =
                new ProgressBar("Export recommendations", totalReviews, 1000,
                        System.out, ProgressBarStyle.ASCII);
        progressBar.start();

//        Map<String, Integer> oneHotIdMap = getOneHot(reviews);

        for (Review review : reviews) {

//            System.out.print("Progress: " + progress + "/" + totalReviews + "\r");
//            progress++;

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
                    row.add(String.valueOf(oneHotIdMap.get(contextId)) + ":" + contextTopics.get(contextColumn));
                }
            }

            bufferedWriter.write(String.join(delimiter, row) + "\n");
            progressBar.step();
        }
        bufferedWriter.flush();
        bufferedWriter.close();
        progressBar.stop();
        System.out.println("\n");
    }


    public static List<Review> getReviewsForRanking(
            List<Review> trainReviews, List<Review> testReviews) {

        Set<Long> testUsersSet = new HashSet<>();
        Set<Long> trainItemsSet = new HashSet<>();

        for (Review review : trainReviews) {
            trainItemsSet.add(review.getItemId());
        }

        for (Review review : testReviews) {
            testUsersSet.add(review.getUserId());
        }

        // Create a map of relevant reviews

        Map<Long, List<Review>> relevantReviewsPerUser = new HashMap<>();
        for (Review review : testReviews) {
            if (review.getRating() >= RELEVANCE_THRESHOLD) {
                if (!relevantReviewsPerUser.containsKey(review.getUserId())) {
                    relevantReviewsPerUser.put(review.getUserId(), new ArrayList<Review>());
                }
                relevantReviewsPerUser.get(review.getUserId()).add(review);
            }
        }



        // Select one review with high rating randomly (preferred item)
        // obtain its context and for all the items in the training set create
        // reviews that have the same context

//        BufferedWriter writer =
//                new BufferedWriter(new FileWriter(filePath));
        List<Review> exportReviews = new ArrayList<>();

        ProgressBar progressBar = new ProgressBar(
                "Export user recommendations", testUsersSet.size(), 1000,
                System.out, ProgressBarStyle.ASCII);
        progressBar.start();

        for (Long user : testUsersSet) {

            List<Review> relevantReviews = relevantReviewsPerUser.get(user);
//            System.out.println("User: " + user);
//            System.out.println(relevantReviews);

            if (relevantReviews == null) {
                continue;
            }

            int reviewIndex = (int) (Math.random() * relevantReviews.size());
            Review relevantReview = relevantReviews.get(reviewIndex);
            Map<String, Double> context = relevantReview.getContext();

            exportReviews.add(relevantReview);
//            writeReviewToFile(relevantReview, writer);

            for (Long item : trainItemsSet) {
                Review review = new Review(user, item);
                review.setContext(context);
                exportReviews.add(review);
//                writeReviewToFile(review, writer);
            }
            progressBar.step();
        }
//        writer.flush();
//        writer.close();
        progressBar.stop();
//        System.out.printf("\n");

        return exportReviews;
    }


    public static Map<String, Integer> getOneHot(Collection<Review> reviews) {

        Map<String, Integer> idMap = new HashMap<>();
        int oneHotId = 0;

        // First we iterate over users
        for (Review review : reviews) {

            String userId = "user_" + String.valueOf(review.getUserId());
            if (!idMap.containsKey(userId)) {
                idMap.put(userId, oneHotId);
                oneHotId++;
            }
        }

        // Then we iterate over items
        for (Review review : reviews) {

            String itemId = "item_" + String.valueOf(review.getItemId());
            if (!idMap.containsKey(itemId)) {
                idMap.put(itemId, oneHotId);
                oneHotId++;
            }
        }

        // Then we iterate over context
        for (Review review : reviews) {

            Map<String, Double> contextTopics = review.getContext();

            if (contextTopics == null) {
                continue;
            }

            for (String contextColumn : contextTopics.keySet()) {

                String contextId = "context_" + contextColumn;

                if (!idMap.containsKey(contextId)) {
                    idMap.put(contextId, oneHotId);
                    oneHotId++;
                }
            }
        }

//        System.out.println(idMap);

        return idMap;
    }

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
        exportRankingPredictionsFile(
                trainReviews, testReviews, filePath, oneHotIdMap, predictionsFile);
    }


    public static void exportRankingPredictionsFile(
            List<Review> trainReviews, List<Review> testReviews,
            String predictionsFilePath, Map<String, Integer> oneHotIdMap, String predictionsFile) throws IOException {

        System.out.println("Export ranking predictions file");

//        List<Review> exportReviews =
//                getReviewsForRanking(trainReviews, testReviews);

        Set<Long> testUsersSet = new HashSet<>();
        Set<Long> trainItemsSet = new HashSet<>();

        for (Review review : trainReviews) {
            trainItemsSet.add(review.getItemId());
        }

        for (Review review : testReviews) {
            testUsersSet.add(review.getUserId());
        }

        // Create a map of relevant reviews

        Map<Long, List<Review>> relevantReviewsPerUser = new HashMap<>();
        for (Review review : testReviews) {
            if (review.getRating() >= RELEVANCE_THRESHOLD) {
                if (!relevantReviewsPerUser.containsKey(review.getUserId())) {
                    relevantReviewsPerUser.put(review.getUserId(), new ArrayList<Review>());
                }
                relevantReviewsPerUser.get(review.getUserId()).add(review);
            }
        }



        // Select one review with high rating randomly (preferred item)
        // obtain its context and for all the items in the training set create
        // reviews that have the same context

        BufferedWriter predictionsWriter =
                new BufferedWriter(new FileWriter(predictionsFile));
        BufferedWriter libfmWriter =
                new BufferedWriter(new FileWriter(predictionsFilePath));
//        List<Review> exportReviews = new ArrayList<>();

        ProgressBar progressBar = new ProgressBar(
                "Export user recommendations", testUsersSet.size(), 1000,
                System.out, ProgressBarStyle.ASCII);
        progressBar.start();

        List<String> firstRow = new ArrayList<>();
        firstRow.add("user");
        firstRow.add("item");
        firstRow.add("predicted_rating");

//        Set<String> contextKeys = trainReviews.get(0).getContext().keySet();
//        for (String context : contextKeys) {
//            firstRow.add(context);
//        }
        String delimiter = "\t";
        // Write the header of the CSV file
        predictionsWriter.write(String.join(delimiter, firstRow) + "\n");

        for (Long user : testUsersSet) {

            List<Review> relevantReviews = relevantReviewsPerUser.get(user);
//            System.out.println("User: " + user);
//            System.out.println(relevantReviews);

            if (relevantReviews == null) {
                progressBar.step();
                continue;
            }

            int reviewIndex = (int) (Math.random() * relevantReviews.size());
            Review relevantReview = relevantReviews.get(reviewIndex);
            Map<String, Double> context = relevantReview.getContext();

//            exportReviews.add(relevantReview);
            writeReviewToFile(relevantReview, predictionsWriter);
            writeLibfmReviewToFile(relevantReview, oneHotIdMap, libfmWriter);

            for (Long item : trainItemsSet) {
                Review review = new Review(user, item);
                review.setContext(context);
//                exportReviews.add(review);
                writeReviewToFile(review, predictionsWriter);
                writeLibfmReviewToFile(review, oneHotIdMap, libfmWriter);
            }
            progressBar.step();
        }
        predictionsWriter.flush();
        predictionsWriter.close();
        libfmWriter.flush();
        libfmWriter.close();
        progressBar.stop();
        System.out.println("\n");


//        for (Review exportReview : exportReviews) {
//            System.out.println(exportReview);
//        }

//        exportRecommendations(exportReviews, predictionsFilePath, oneHotIdMap);
    }


    public static void writeReviewToFile(Review review, BufferedWriter writer)
            throws IOException {

        String delimiter = "\t";
        List<String> row = new ArrayList<>();
        row.add(String.valueOf(review.getUserId()));
        row.add(String.valueOf(review.getItemId()));
        row.add(Double.toString(review.getPredictedRating()));

//        for (Double context : review.getContext().values()) {
//            row.add(Double.toString(context));
//        }

        writer.write(String.join(delimiter, row) + "\n");
    }


    private static void writeLibfmReviewToFile(
            Review review, Map<String, Integer> oneHotIdMap,
            BufferedWriter bufferedWriter) throws IOException {

        String userId = "user_" + String.valueOf(review.getUserId());
        String itemId = "item_" + String.valueOf(review.getItemId());
        String delimiter = " ";

        Map<String, Double> contextTopics = review.getContext();

        List<String> row = new ArrayList<>();
        row.add(String.valueOf(review.getRating()));
        row.add(String.valueOf(oneHotIdMap.get(userId)) + ":1");
        row.add(String.valueOf(oneHotIdMap.get(itemId)) + ":1");

        if (contextTopics != null) {
            for (String contextColumn : contextTopics.keySet()) {

                String contextId = "context_" + contextColumn;
                row.add(String.valueOf(oneHotIdMap.get(contextId)) + ":" + contextTopics.get(contextColumn));
            }
        }

        bufferedWriter.write(String.join(delimiter, row) + "\n");
    }
}
