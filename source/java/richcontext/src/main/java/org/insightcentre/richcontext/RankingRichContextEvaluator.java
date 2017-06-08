package org.insightcentre.richcontext;

import com.opencsv.CSVWriter;
import net.recommenders.rival.core.*;
import net.recommenders.rival.evaluation.metric.error.MAE;
import net.recommenders.rival.evaluation.metric.error.RMSE;
import net.recommenders.rival.evaluation.metric.ranking.NDCG;
import net.recommenders.rival.evaluation.metric.ranking.Precision;
import net.recommenders.rival.evaluation.metric.ranking.Recall;
import net.recommenders.rival.evaluation.strategy.EvaluationStrategy;
import net.recommenders.rival.evaluation.strategy.RelPlusN;
import net.recommenders.rival.split.splitter.CrossValidationSplitter;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Created by fpena on 23/03/2017.
 */
public class RankingRichContextEvaluator {

    /**
     * Default number of folds.
     */
    public static final int NUM_FOLDS = 5;
    /**
     * Default neighbohood size.
     */
    public static final int NEIGHBOURHOOD_SIZE = 50;
    /**
     * Default cutoff for evaluation metrics.
     */
    public static final int AT = 10;
    /**
     * Default cutoff for evaluation metrics.
     */
    public static final int ADDITIONAL_ITEMS = 1000;
    /**
     * Default relevance threshold.
     */
    public static final double RELEVANCE_THRESHOLD = 5.0;
    /**
     * Default seed.
     */
    public static final long SEED = 2048L;

    public static final Strategy STRATEGY = Strategy.REL_PLUS_N;

    public static final ContextFormat CONTEXT_FORMAT = ContextFormat.CONTEXT_TOPIC_WEIGHTS;

    public static final Dataset DATASET = Dataset.YELP_RESTAURANT;


    private final int numFolds;
    private final int at;
    private final int additionaItems;
    private final double relevanceThreshold;
    private final long seed;
    private final Strategy strategy;
    private final ContextFormat contextFormat;
    private final Dataset dataset;
//    private final String outputFolder;


    private Map<String, Review> reviewsMap;
    private String ratingsFolderPath;
    private String jsonRatingsFile;
    private int numUsers;
    private int numItems;


    private enum Strategy {
        ALL_ITEMS,
        REL_PLUS_N,
        TEST_ITEMS,
        TRAIN_ITEMS,
        USER_TEST
    }

    private enum ContextFormat {
        NO_CONTEXT,
        CONTEXT_TOPIC_WEIGHTS,
        TOP_WORDS,
        PREDEFINED_CONTEXT,
        TOPIC_PREDEFINED_CONTEXT
    }

    private enum Dataset {
        YELP_HOTEL,
        YELP_RESTAURANT,
        FOURCITY_HOTEL,
    }



    public RankingRichContextEvaluator(String jsonRatingsFile) throws IOException {
        this.jsonRatingsFile = jsonRatingsFile;

        Properties properties = Properties.loadProperties();
        numFolds = properties.getCrossValidationNumFolds();
        at = properties.getTopN();
        relevanceThreshold = properties.getRelevanceThreshold();
        seed = properties.getSeed();
        strategy = Strategy.valueOf((properties.getStrategy().toUpperCase(Locale.ENGLISH)));
        additionaItems = properties.getTopnNumItems();
        contextFormat = ContextFormat.valueOf(properties.getContextFormat().toUpperCase(Locale.ENGLISH));
        dataset = RankingRichContextEvaluator.Dataset.valueOf(properties.getDataset().toUpperCase(Locale.ENGLISH));

        
        

        init();
    }

    public static void main(final String[] args) throws IOException, InterruptedException, ParseException {

        // create Options object
        Options options = new Options();

        // add t option
        options.addOption("d", false, "The folder containing the data file");
        options.addOption("o", true, "The folder containing the output file");

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse( options, args);
//        String cacheFolder = cmd.getOptionValue("d") == null
//                ? cmd.getOptionValue("d")
//                : "/Users/fpena/UCC/Thesis/datasets/context/stuff/cache_context/";
//        String outputFolder = cmd.getOptionValue("o");

        long startTime = System.currentTimeMillis();

//        String folder = "data/rich-context/";
//        String modelPath = folder + "model/";
//        String recPath = folder + "recommendations/";
        int nFolds = NUM_FOLDS;
//        String dataFile = folder + "yelp_hotel.json";
//        String algorithm = "GlobalAvg";
//        String algorithm = "UserAvg";
//        String algorithm = "ItemAvg";
//        String algorithm = "UserItemAvg";

//        String workingPath;

//        workingPath = "/Users/fpena/UCC/Thesis/datasets/context/stuff/cache_context/" +
//                "carskit/yelp_hotel_carskit_ratings_ensemble_numtopics-30" +
//                "_iterations-100_passes-10_targetreview-specific_" +
//                "normalized_ck-no_context_lang-en_bow-NN_" +
//                "document_level-review_targettype-context_" +
//                "min_item_reviews-10/";

        String cacheFolder =
                "/Users/fpena/UCC/Thesis/datasets/context/stuff/cache_context/";
        String jsonFile;
//        jsonFile = cacheFolder + "yelp_hotel_recsys_contextual_records_ensemble_" +
//                "numtopics-10_iterations-100_passes-10_targetreview-specific_" +
//                "normalized_lang-en_bow-NN_document_level-review_" +
//                "targettype-context_min_item_reviews-10.json";
//        jsonFile = cacheFolder + "yelp_restaurant_recsys_contextual_records_ensemble_" +
//                "numtopics-50_iterations-100_passes-10_targetreview-specific_" +
//                "normalized_lang-en_bow-NN_document_level-review_" +
//                "targettype-context_min_item_reviews-10.json";

//        jsonFile = cacheFolder + "fourcity_hotel_recsys_contextual_records_ensemble_" +
//                "numtopics-10_iterations-100_passes-10_targetreview-specific_" +
//                "normalized_lang-en_bow-NN_document_level-review_" +
//                "targettype-context_min_item_reviews-10.json";

        jsonFile = cacheFolder + DATASET.toString().toLowerCase() +
                "_recsys_formatted_context_records_ensemble_" +
                "numtopics-10_iterations-100_passes-10_targetreview-specific_" +
                "normalized_contextformat-" + CONTEXT_FORMAT.toString().toLowerCase()
                + "_lang-en_bow-NN_document_level-review_targettype-context_" +
                "min_item_reviews-10.json";

        // Non-contextual files
//        jsonFile = cacheFolder + "yelp_hotel_recsys_contextual_records_" +
//                "lang-en_bow-NN_document_level-review_targettype-context_" +
//                "min_item_reviews-10.json";



//        workingPath = "/Users/fpena/tmp/CARSKit/context-aware_data_sets/yelp_hotel/";

        RankingRichContextEvaluator evaluator = new RankingRichContextEvaluator(jsonFile);
        evaluator.prepareSplits(nFolds);
//        evaluator.transformSplitsToCarskit(NUM_FOLDS);
//        evaluator.transformSplitsToLibfm(NUM_FOLDS);
        evaluator.parseRecommendationResultsLibfm(NUM_FOLDS);
        evaluator.prepareStrategy(NUM_FOLDS, "libfm");
        Map<String, String> results = evaluator.evaluate(NUM_FOLDS, "libfm");

        List<Map<String, String>> resultsList = new ArrayList<>();
        resultsList.add(results);
        writeResultsToFile(resultsList);

        String[] algorithms = {
//                "GlobalAvg",
//                "UserAvg",
//                "ItemAvg",
//                "UserItemAvg",
//                "SlopeOne",
//                "PMF",
//                "BPMF",
//                "BiasedMF",
//                "NMF",
//                "CAMF_CI", "CAMF_CU",
//                "CAMF_CUCI",
//                "SLIM",
//                "BPR",
//                "LRMF",
//                "CSLIM_C", "CSLIM_CI",
//                "CSLIM_CU",
        };

        int progress = 1;
        for (String algorithm : algorithms) {

            System.out.println(
                    "\n\nProgress: " + progress + "/" + algorithms.length);
            evaluator.postProcess(NUM_FOLDS, algorithm);
            progress++;
        }


//        RatingContextEvaluator evaluator = new RatingContextEvaluator("GlobalAvg", workingPath);

//        String fileName = getRecommendationsFileName(workingPath, "GlobalAvg", 1, -10);
//        System.out.println(fileName);

        long endTime   = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Running time: " + (totalTime/1000));
    }


    private void init() {

        File jsonFile =  new File(jsonRatingsFile);
        String jsonFileName = jsonFile.getName();
        String jsonFileParentFolder = jsonFile.getParent();
        String rivalFolderPath = jsonFileParentFolder + "/rival/";

        File rivalDir = new File(rivalFolderPath);
        if (!rivalDir.exists()) {
            if (!rivalDir.mkdir()) {
                System.err.println("Directory " + rivalDir + " could not be created");
                return;
            }
        }

        // We strip the extension of the file name to create a new folder with
        // an unique name
        if (jsonFileName.indexOf(".") > 0) {
            jsonFileName = jsonFileName.substring(0, jsonFileName.lastIndexOf("."));
        }

        ratingsFolderPath = rivalFolderPath + jsonFileName + "/";

        File ratingsDir = new File(ratingsFolderPath);
        if (!ratingsDir.exists()) {
            if (!ratingsDir.mkdir()) {
                System.err.println("Directory " + ratingsDir + " could not be created");
                return;
            }
        }

        System.out.println("File name: " + jsonFileName);
        System.out.println("Parent folder: " + jsonFileParentFolder);
        System.out.println("Ratings folder: " + ratingsFolderPath);
    }

    /**
     * Downloads a dataset and stores the splits generated from it.
     *
     * @param nFolds number of folds
     */
    public void prepareSplits(final int nFolds) {

        String dataFile = jsonRatingsFile;
        boolean perUser = false;
        long seed = SEED;
        JsonParser parser = new JsonParser();

        DataModelIF<Long, Long> data = null;
        try {
            data = parser.parseData(new File(dataFile));
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Build reviews map
        this.reviewsMap = new HashMap<>();
        Set<Long> itemsSet = new HashSet<>();
        Set<Long> usersSet = new HashSet<>();
        for (Review review : parser.getReviews()) {
            reviewsMap.put(review.getUser_item_key(), review);
            itemsSet.add(review.getItemId());
            usersSet.add(review.getUserId());
        }
        this.numItems = itemsSet.size();
        this.numUsers = usersSet.size();
        System.out.println("Num items: " + this.numItems);
        System.out.println("Num users: " + this.numUsers);


        DataModelIF<Long, Long>[] splits =
                new CrossValidationSplitter<Long, Long>(
                        nFolds, perUser, seed).split(data);
        File dir = new File(ratingsFolderPath);
        if (!dir.exists()) {
            if (!dir.mkdir()) {
                System.err.println("Directory " + dir + " could not be created");
                return;
            }
        }
        for (int i = 0; i < splits.length / 2; i++) {

            String foldPath = ratingsFolderPath + "fold_" + i + "/";
            File foldDir = new File(foldPath);
            if (!foldDir.exists()) {
                if (!foldDir.mkdir()) {
                    System.err.println("Directory " + foldDir + " could not be created");
                    return;
                }
            }

            DataModelIF<Long, Long> training = splits[2 * i];
            DataModelIF<Long, Long> test = splits[2 * i + 1];
            String trainingFile = foldPath + "train.csv";
            String testFile = foldPath + "test.csv";
            boolean overwrite = true;
            try {
                DataModelUtils.saveDataModel(training, trainingFile, overwrite, "\t");
                DataModelUtils.saveDataModel(test, testFile, overwrite, "\t");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
    }


    public void transformSplitToCarskit(int fold) throws IOException {

        String foldPath = ratingsFolderPath + "fold_" + fold + "/";
        String trainFile = foldPath + "train.csv";
        String testFile = foldPath + "test.csv";
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

        String carskitTrainFile = foldPath + "carskit_train.csv";
        String carskitTestFile = foldPath + "carskit_test.csv";

//        CarskitExporter.exportWithoutContext(
//                completeTrainReviews, carskitTrainFile);
//        CarskitExporter.exportWithoutContext(
//                completeTestReviews, carskitTestFile);
        CarskitExporter.exportWithContext(
                completeTrainReviews, carskitTrainFile);
        CarskitExporter.exportWithContext(
                completeTestReviews, carskitTestFile);

        String workspacePath = foldPath + "CARSKit.Workspace/";
        File workspaceDir = new File(workspacePath);
        if (!workspaceDir.exists()) {
            if (!workspaceDir.mkdir()) {
                System.err.println("Directory " + workspaceDir + " could not be created");
                return;
            }
        }

        String ratingsBinaryPath = workspacePath + "ratings_binary.txt";

        Files.copy(
                new File(carskitTrainFile).toPath(),
                new File(ratingsBinaryPath).toPath(),
                StandardCopyOption.REPLACE_EXISTING);
    }


    public void transformSplitsToCarskit(int numFolds) throws IOException {

        for (int fold = 0; fold < numFolds; fold++) {
            transformSplitToCarskit(fold);
        }
    }


    public void transformSplitToLibfm(int fold) throws IOException {

        String foldPath = ratingsFolderPath + "fold_" + fold + "/";
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

        String libfmTrainFile = foldPath + "libfm_train.libfm";
        String libfmTestFile = foldPath + "libfm_test.libfm";
        String libfmPredictionsFile = foldPath + "libfm_preds.libfm";

        Map<String, Integer> oneHotIdMap =
                LibfmExporter.getOneHot(reviewsMap.values());

        LibfmExporter.exportRecommendations(
                completeTrainReviews, libfmTrainFile, oneHotIdMap);
        LibfmExporter.exportRecommendations(
                completeTestReviews, libfmTestFile, oneHotIdMap);

        List<Review> reviewsForRanking = LibfmExporter.getReviewsForRanking(
                completeTrainReviews, completeTestReviews, oneHotIdMap);
        CarskitExporter.exportContextRecommendationsToCsv(
                reviewsForRanking, predictionsFile);
        LibfmExporter.exportRankingPredictionsFile(
                completeTrainReviews, completeTestReviews, libfmPredictionsFile,
                oneHotIdMap
        );
    }


    public void transformSplitsToLibfm(int numFolds) throws IOException {

        for (int fold = 0; fold < numFolds; fold++) {
            transformSplitToLibfm(fold);
        }
    }


    public void parseRecommendationResults(int numFolds, String algorithm) throws IOException, InterruptedException {

        System.out.println("Parse Recommendation Results");

        for (int fold = 0; fold < numFolds; fold++) {


            // Execute the recommender

//            String configFile = ratingsFolderPath + algorithm + "_" + fold + ".conf";
//            CarskitCaller.run(configFile);


            // Collect the results from the recommender
            String recommendationsFile = getRecommendationsFileName(
                    ratingsFolderPath, algorithm, fold, AT);

            List<Review> recommendations = (AT < 1) ?
                    CarskitResultsParser.parseRatingResults(recommendationsFile) :
                    CarskitResultsParser.parseRankingResults(recommendationsFile);

            String rivalRecommendationsFile =
                    ratingsFolderPath + "fold_" + fold + "/recs_" + algorithm + ".csv";
            System.out.println("Recommendations file name: " + rivalRecommendationsFile);
            CarskitExporter.exportRecommendationsToCsv(
                    recommendations, rivalRecommendationsFile);
        }
    }


    public void parseRecommendationResultsLibfm(int numFolds) throws IOException, InterruptedException {

        System.out.println("Parse Recommendation Results");

        for (int fold = 0; fold < numFolds; fold++) {

            // Collect the results from the recommender
            String foldPath = ratingsFolderPath + "fold_" + fold + "/";
            String testFile = foldPath + "test.csv";
            String predictionsFile = foldPath + "predictions.csv";
            String libfmResultsFile = foldPath + "libfm_predictions.txt";

            // TODO: Update here to include context, create a parseContextRatingResutls
            List<Review> recommendations =
                    LibfmResultsParser.parseContextRatingResults(
                            predictionsFile, libfmResultsFile);

            String rivalRecommendationsFile =
                    ratingsFolderPath + "fold_" + fold + "/recs_libfm" + ".csv";
            System.out.println("Recommendations file name: " + rivalRecommendationsFile);
//            CarskitExporter.exportContextRecommendationsToCsv(
//                    recommendations, rivalRecommendationsFile);
            CarskitExporter.exportRecommendationsToCsv(
                    recommendations, rivalRecommendationsFile);
        }
    }


    public void prepareStrategy(final int nFolds, String algorithm) {

        System.out.println("Prepare Strategy");

        for (int i = 0; i < nFolds; i++) {

            String foldPath = ratingsFolderPath + "fold_" + i + "/";
            File trainingFile = new File(foldPath + "train.csv");
            File testFile = new File(foldPath + "test.csv");
            File recFile = new File(foldPath + "recs_" + algorithm + ".csv");
            DataModelIF<Long, Long> trainingModel;
            DataModelIF<Long, Long> testModel;
            NewContextDataModel<Long, Long> recModel;
//            DataModelIF<Long, Long> recModel;
            try {
                trainingModel = new SimpleParser().parseData(trainingFile);
                testModel = new SimpleParser().parseData(testFile);

                // TODO: Create a parser for context files
                recModel = new ContextParser().parseData2(recFile, "\t");
//                recModel = new SimpleParser().parseData(recFile);
            } catch (IOException e) {
                e.printStackTrace();
                return;
            }

            System.out.println("Recommendation model num users: " + recModel.getNumUsers());
            System.out.println("Recommendation model num items: " + recModel.getNumItems());
            System.out.println("Recommendation model num predictions: " + recModel.getUserContextItemPreferences().size());
//            System.out.println("Recommendation model num predictions: " + recModel.getUserItemPreferences().size());

            Double threshold = RELEVANCE_THRESHOLD;
            EvaluationStrategy<Long, Long> evaluationStrategy =
                    new RelPlusN(trainingModel, testModel, ADDITIONAL_ITEMS, threshold, SEED);

            // TODO: Change the type of modelToEval to ContextDataModel, so that it is possible to have dupliceates of same user-item pairs with different context
            NewContextDataModel<Long, Long> modelToEval = new NewContextDataModel<>();
//            DataModelIF<Long, Long> modelToEval = DataModelFactory.getDefaultModel();
            for (Long user : recModel.getUsers()) {

                Map<Map<String, Double>, Map<Long, Double>> userContextPreferences =
                        recModel.getUserContextItemPreferences().get(user);
                Map<String, Double> context = userContextPreferences.keySet().iterator().next();
                Map<Long, Double> itemPreferences = userContextPreferences.get(context);
//
                if (userContextPreferences.size() != 1) {
                    System.out.println("User context Preferences size: " + userContextPreferences.size());
                }

//                Map<Long, Double> itemPreferences = recModel.getUserItemPreferences().get(user);

                for (Long item : evaluationStrategy.getCandidateItemsToRank(user)) {

                    if (itemPreferences.containsKey(item)) {
                        modelToEval.addPreference(user, item, context, itemPreferences.get(item));
//                        modelToEval.addPreference(user, item, itemPreferences.get(item));
                    }
                }
//                for (Long item : evaluationStrategy.getCandidateItemsToRank(user)) {
//                    if (recModel.getUserItemPreferences().get(user).containsKey(item)) {
//                        modelToEval.addPreference(user, item, recModel.getUserItemPreferences().get(user).get(item));
//                    }
//                }
            }
            try {
                modelToEval.saveDataModel(foldPath + "strategymodel_" + algorithm + "_" + STRATEGY.toString() + ".csv", true, "\t");
//                DataModelUtils.saveDataModel(modelToEval, foldPath + "strategymodel_" + algorithm + "_" + STRATEGY.toString() + ".csv", true, "\t");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
    }


    /**
     * Evaluates the recommendations generated in previous steps.
     *
     * @param nFolds number of folds
     */
    public Map<String, String> evaluate(final int nFolds, String algorithm) {

        System.out.println("Evaluate");

        double ndcgRes = 0.0;
        double recallRes = 0.0;
        double precisionRes = 0.0;
        double rmseRes = 0.0;
        double maeRes = 0.0;
        Map<String, String> results =  new HashMap<>();
        for (int i = 0; i < nFolds; i++) {
            String foldPath = ratingsFolderPath + "fold_" + i + "/";
            File testFile = new File(foldPath + "test.csv");
            File strategyFile = new File(foldPath + "strategymodel_" + algorithm + "_" + STRATEGY.toString() + ".csv");
            DataModelIF<Long, Long> testModel = null;
            DataModelIF<Long, Long> recModel = null;
            try {
                testModel = new SimpleParser().parseData(testFile);
                recModel = new SimpleParser().parseData(strategyFile);
            } catch (IOException e) {
                e.printStackTrace();
            }
            NDCG<Long, Long> ndcg = new NDCG<>(recModel, testModel, new int[]{AT});
            ndcg.compute();
            ndcgRes += ndcg.getValueAt(AT);
            results.put("fold_" + i + "_ndcg", String.valueOf(ndcg.getValueAt(AT)));

            Recall<Long, Long> recall = new Recall<>(recModel, testModel, RELEVANCE_THRESHOLD, new int[]{AT});
            recall.compute();
            recallRes += recall.getValueAt(AT);
            results.put("fold_" + i + "_recall", String.valueOf(ndcg.getValueAt(AT)));

            RMSE<Long, Long> rmse = new RMSE<>(recModel, testModel);
            rmse.compute();
            rmseRes += rmse.getValue();
            results.put("fold_" + i + "_rmse", String.valueOf(ndcg.getValue()));

            MAE<Long, Long> mae = new MAE<>(recModel, testModel);
            mae.compute();
            maeRes += mae.getValue();
            results.put("fold_" + i + "_mae", String.valueOf(ndcg.getValue()));

            Precision<Long, Long> precision = new Precision<>(recModel, testModel, RELEVANCE_THRESHOLD, new int[]{AT});
            precision.compute();
            precisionRes += precision.getValueAt(AT);
            results.put("fold_" + i + "_precision", String.valueOf(ndcg.getValueAt(AT)));
        }

        results.put("Algorithm", algorithm);
        results.put("Strategy", STRATEGY.toString());
        results.put("Context_Format", CONTEXT_FORMAT.toString());
        results.put("NDCG@" + AT, String.valueOf(ndcgRes / nFolds));
        results.put("Precision@" + AT, String.valueOf(precisionRes / nFolds));
        results.put("Recall@" + AT, String.valueOf(recallRes / nFolds));
        results.put("RMSE", String.valueOf(rmseRes / nFolds));
        results.put("MAE", String.valueOf(maeRes / nFolds));

        System.out.println("Algorithm: " + algorithm);
        System.out.println("Strategy: " + STRATEGY.toString());
        System.out.println("Context_Format: " + CONTEXT_FORMAT);
        System.out.println("NDCG@" + AT + ": " + ndcgRes / nFolds);
        System.out.println("Precision@" + AT + ": " + precisionRes / nFolds);
        System.out.println("Recall@" + AT + ": " + recallRes / nFolds);
        System.out.println("RMSE: " + rmseRes / nFolds);
        System.out.println("MAE: " + maeRes / nFolds);
//        System.out.println("P@" + AT + ": " + precisionRes / nFolds);

        return results;
    }


    public void postProcess(int numFolds, String algorithm)
            throws IOException, InterruptedException {

        List<Map<String, String>> resultsList = new ArrayList<>();

//        RatingContextEvaluator evaluator = new RatingContextEvaluator(jsonFile);
        parseRecommendationResults(numFolds, algorithm);
        prepareStrategy(numFolds, algorithm);
        resultsList.add(evaluate(numFolds, algorithm));
        writeResultsToFile(resultsList);

    }


    private static void writeResultsToFile(
            List<Map<String, String>> resultsList) throws IOException {

        String[] headers = {
                "Algorithm",
                "Strategy",
                "Context_Format",
                "NDCG@" + AT,
                "Precision@" + AT,
                "Recall@" + AT,
                "RMSE",
                "MAE",
                "fold_0_ndcg",
                "fold_1_ndcg",
                "fold_2_ndcg",
                "fold_3_ndcg",
                "fold_4_ndcg",
                "fold_0_precision",
                "fold_1_precision",
                "fold_2_precision",
                "fold_3_precision",
                "fold_4_precision",
                "fold_0_recall",
                "fold_1_recall",
                "fold_2_recall",
                "fold_3_recall",
                "fold_4_recall",
                "fold_0_rmse",
                "fold_1_rmse",
                "fold_2_rmse",
                "fold_3_rmse",
                "fold_4_rmse",
                "fold_0_mae",
                "fold_1_mae",
                "fold_2_mae",
                "fold_3_mae",
                "fold_4_mae",
        };

        String dataset = DATASET.toString().toLowerCase();
        String fileName = "/Users/fpena/tmp/rival_" + dataset + "_results-folds.csv";

        File resultsFile = new File(fileName);
        boolean fileExists = resultsFile.exists();
        CSVWriter writer = new CSVWriter(
                new FileWriter(resultsFile, true),
                ',', CSVWriter.NO_QUOTE_CHARACTER);

        if (!fileExists) {
            writer.writeNext(headers);
        }

        for (Map<String, String> results : resultsList) {
            String[] row = new String[headers.length];

            for (int i = 0; i < headers.length; i++) {
                row[i] = results.get(headers[i]);
            }
            writer.writeNext(row);
        }
        writer.close();

    }



    public String getRecommendationsFileName(
            String workingPath, String algorithm, int foldIndex, int topN) {

//        String dataFile = jsonRatingsFile;
//        JsonParser parser = new JsonParser();
//        try {
//            parser.parseData(new File(dataFile));
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//
//        Set<Long> itemsSet = new HashSet<>();
//        Set<Long> usersSet = new HashSet<>();
//        for (Review review : parser.getReviews()) {
//            itemsSet.add(review.getItemId());
//            usersSet.add(review.getUserId());
//        }
//        this.numItems = itemsSet.size();
//        this.numUsers = usersSet.size();
//        System.out.println("Num items: " + this.numItems);
//        System.out.println("Num users: " + this.numUsers);


        String filePath;
        String carskitWorkingPath =
                workingPath + "fold_" + foldIndex + "/CARSKit.Workspace/";
//        String foldInfo = foldIndex > 0 ? " fold [" + foldIndex + "]" : "";

        // This means that we are going to generate a rating prediction file name
        if (topN < 1) {
            filePath =
                    carskitWorkingPath + algorithm+ "-rating-predictions.txt";
        }
        else {
            filePath = carskitWorkingPath + String.format(
                    "%s-top-%d-items.txt", algorithm, this.numItems);
        }

        return filePath;
    }
}
