package org.insightcentre.richcontext;

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
 * Created by fpena on 17/09/2018.
 */
public abstract class ReviewsExporter {


    public static final double RELEVANCE_THRESHOLD = 5.0;
    protected final String DELIMITER = getDelimiter();



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


    public static void writeReviewToFile(Review review, BufferedWriter writer)
            throws IOException {

        String delimiter = "\t";
        List<String> row = new ArrayList<>();
        row.add(String.valueOf(review.getUserId()));
        row.add(String.valueOf(review.getItemId()));
        row.add(Double.toString(review.getPredictedRating()));

        writer.write(String.join(delimiter, row) + "\n");
    }

    public void exportRecommendations(
            List<Review> reviews, String filePath,
            Map<String, Integer> oneHotIdMap) throws IOException {

        BufferedWriter bufferedWriter =
                new BufferedWriter(new FileWriter(filePath));

        List<String> contextKeys = new ArrayList<>(reviews.get(0).getContext().keySet());
        writeHeader(contextKeys, bufferedWriter);

        for (Review review : reviews) {

            writeLibfmReviewToFile(review, oneHotIdMap, bufferedWriter);
        }
        bufferedWriter.flush();
        bufferedWriter.close();
        System.out.println("\n");
    }


    public void exportRankingPredictionsFile(
            List<Review> trainReviews, List<Review> testReviews,
            String predictionsFilePath, Map<String, Integer> oneHotIdMap, String predictionsFile) throws IOException {

        System.out.println("Export ranking predictions file");

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

        for (Long user : testUsersSet) {

            List<Review> relevantReviews = relevantReviewsPerUser.get(user);

            if (relevantReviews == null) {
                continue;
            }

            int reviewIndex = (int) (Math.random() * relevantReviews.size());
            Review relevantReview = relevantReviews.get(reviewIndex);
            Map<String, Double> context = relevantReview.getContext();

            writeReviewToFile(relevantReview, predictionsWriter);
            writeLibfmReviewToFile(relevantReview, oneHotIdMap, libfmWriter);

            for (Long item : trainItemsSet) {
                Review review = new Review(user, item);
                // In case we have a duplicate (user,item) pair, we skip it
                if (user == relevantReview.getUserId() && item == relevantReview.getItemId()) {
                    continue;
                }
                review.setContext(context);
                writeReviewToFile(review, predictionsWriter);
                writeLibfmReviewToFile(review, oneHotIdMap, libfmWriter);
            }
        }
        predictionsWriter.flush();
        predictionsWriter.close();
        libfmWriter.flush();
        libfmWriter.close();
        System.out.println("\n");
    }

    protected abstract String getDelimiter();

    protected abstract void writeHeader(
            List<String> contextKeys, BufferedWriter writer) throws IOException;

    protected abstract void writeLibfmReviewToFile(
            Review review, Map<String, Integer> oneHotIdMap,
            BufferedWriter bufferedWriter) throws IOException;

}
