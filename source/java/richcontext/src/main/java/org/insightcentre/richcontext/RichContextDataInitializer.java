package org.insightcentre.richcontext;

import net.recommenders.rival.core.DataModelIF;
import net.recommenders.rival.core.DataModelUtils;
import net.recommenders.rival.split.splitter.CrossValidationSplitter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Created by fpena on 05/09/2017.
 */
public class RichContextDataInitializer {


    private final int numFolds;
    private final long seed;
    private final RichContextResultsProcessor.Strategy strategy;


    private Map<String, Review> reviewsMap;
    private String ratingsFolderPath;
    private String jsonRatingsFile;


    public RichContextDataInitializer(
            String cacheFolder, String propertiesFile, Integer paramNumTopics,
            String itemType)
            throws IOException {

        Properties properties = Properties.loadProperties(propertiesFile);
        numFolds = properties.getCrossValidationNumFolds();
        seed = properties.getSeed();
        strategy = RichContextResultsProcessor.Strategy.valueOf(
                (properties.getStrategy().toUpperCase(Locale.ENGLISH)));
        RichContextResultsProcessor.ContextFormat contextFormat = RichContextResultsProcessor.ContextFormat.valueOf(
                properties.getContextFormat().toUpperCase(Locale.ENGLISH));
        RichContextResultsProcessor.Dataset dataset = RichContextResultsProcessor.Dataset.valueOf(
                properties.getDataset().toUpperCase(Locale.ENGLISH));
        int numTopics = (paramNumTopics == null) ?
                properties.getNumTopics() :
                paramNumTopics;

        jsonRatingsFile = cacheFolder + itemType.toLowerCase() +
                "_recsys_formatted_context_records_ensemble_" +
                "numtopics-" + numTopics + "_iterations-100_passes-10_targetreview-specific_" +
                "normalized_contextformat-" + contextFormat.toString().toLowerCase()
                + "_lang-en_bow-NN_document_level-review_targettype-context_" +
                "min_item_reviews-10.json";

        this.ratingsFolderPath = Utils.getRatingsFolderPath(jsonRatingsFile);
    }


    /**
     * Creates the cross-validation splits
     */
    private void prepareSplits() throws IOException {

        String dataFile = jsonRatingsFile;
        boolean perUser = false;
        JsonParser parser = new JsonParser();

        DataModelIF<Long, Long> data = parser.parseData(new File(dataFile));

        // Build reviews map
        this.reviewsMap = new HashMap<>();
        Set<Long> itemsSet = new HashSet<>();
        Set<Long> usersSet = new HashSet<>();
        for (Review review : parser.getReviews()) {
            reviewsMap.put(review.getUser_item_key(), review);
            itemsSet.add(review.getItemId());
            usersSet.add(review.getUserId());
        }
        int numItems = itemsSet.size();
        int numUsers = usersSet.size();
        System.out.println("Num items: " + numItems);
        System.out.println("Num users: " + numUsers);


        DataModelIF<Long, Long>[] splits =
                new CrossValidationSplitter<Long, Long>(
                        numFolds, perUser, seed).split(data);
        File dir = new File(ratingsFolderPath);
        if (!dir.exists()) {
            if (!dir.mkdir()) {
                System.out.println("Directory " + dir + " could not be created");
                return;
            }
        }
        for (int i = 0; i < splits.length / 2; i++) {

            String foldPath = ratingsFolderPath + "fold_" + i + "/";
            File foldDir = new File(foldPath);
            if (!foldDir.exists()) {
                if (!foldDir.mkdir()) {
                    System.out.println("Directory " + foldDir + " could not be created");
                    return;
                }
            }

            DataModelIF<Long, Long> training = splits[2 * i];
            DataModelIF<Long, Long> test = splits[2 * i + 1];

            String trainingFile = foldPath + "train.csv";
            String testFile = foldPath + "test.csv";
            boolean overwrite = true;
            DataModelUtils.saveDataModel(training, trainingFile, overwrite, "\t");
            DataModelUtils.saveDataModel(test, testFile, overwrite, "\t");
        }
    }


    /**
     * Transform the split given by {@code fold} into the format required by
     * LibFM
     *
     * @param fold the cross-validation fold
     */
    private void transformSplitToLibfm(int fold, RecommenderLibrary library)
            throws IOException {

        System.out.println("Transform split " + fold + " to LibFM");
        boolean overwrite = false;

        String libraryName = library.getName();
        String extension = library.getExtension();
        ReviewsExporter exporter = library.getReviewsExporter();

        String foldPath = ratingsFolderPath + "fold_" + fold + "/";
        String libfmTrainFile = foldPath + libraryName + "_train." + extension;
//        String libfmTestFile = foldPath + "libfm_test.libfm";
        String libfmPredictionsFile = foldPath + libraryName + "_predictions_" +
                strategy.getPredictionType() + "." + extension;

        if (new File(libfmTrainFile).exists() &&
                new File(libfmPredictionsFile).exists() && !overwrite) {
            System.out.println(
                    "LibFM files for split " + fold + " already exist");
            return;
        }

        String trainFile = foldPath + "train.csv";
        String testFile = foldPath + "test.csv";
        String predictionsFile = foldPath + "predictions.csv";
        List<Review> incompleteTrainReviews = ReviewCsvDao.readCsvFile(trainFile);
        List<Review> incompleteTestReviews = ReviewCsvDao.readCsvFile(testFile);

        List<Review> completeTrainReviews = new ArrayList<>();
        List<Review> completeTestReviews = new ArrayList<>();

        for (Review review : incompleteTrainReviews) {
            Review completeReview = reviewsMap.get(review.getUser_item_key());
            completeTrainReviews.add(completeReview);
        }
        for (Review review : incompleteTestReviews) {
            Review completeReview = reviewsMap.get(review.getUser_item_key());
            completeTestReviews.add(completeReview);
        }

        Map<String, Integer> oneHotIdMap =
                LibfmExporter.getOneHot(reviewsMap.values());

//        LibfmExporter exporter = new LibfmExporter();
        exporter.exportRecommendations(
                completeTrainReviews, libfmTrainFile, oneHotIdMap);

        switch (strategy) {
            case TEST_ITEMS:
            case USER_TEST:
                exporter.exportRecommendations(
                        completeTestReviews, libfmPredictionsFile, oneHotIdMap);
                break;
            case REL_PLUS_N: {
                exporter.exportRankingPredictionsFile(
                        completeTrainReviews, completeTestReviews,
                        libfmPredictionsFile, oneHotIdMap, predictionsFile
                );
                break;
            }
            default:
                throw new UnsupportedOperationException(
                        strategy.toString() + " evaluation Strategy not supported");
        }

    }


    /**
     * Transform the cross-validation splits into the format required by
     * LibFM
     */
    private void transformSplitsToLibfm(RecommenderLibrary library)
            throws IOException {

        for (int fold = 0; fold < numFolds; fold++) {
            transformSplitToLibfm(fold, library);
        }
    }


    /**
     * Runs the whole data preparation cycle leaving the data ready for the
     * recommender to make predictions.
     *
     * @param cacheFolder the folder that contains the dataset to be evaluated
     * @param propertiesFile the file that contains the hyperparameters for the
     * recommender
     */
    public static void prepareLibfm(
            String cacheFolder, String propertiesFile, Integer numTopics,
            String itemType, RecommenderLibrary library)
            throws IOException {

        RichContextDataInitializer evaluator = new RichContextDataInitializer(
                cacheFolder, propertiesFile, numTopics, itemType);
        evaluator.prepareSplits();
        evaluator.transformSplitsToLibfm(library);
    }
}
