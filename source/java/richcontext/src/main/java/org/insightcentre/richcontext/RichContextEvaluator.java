package org.insightcentre.richcontext;

import com.opencsv.CSVWriter;
import net.recommenders.rival.core.DataModel;
import net.recommenders.rival.core.DataModelFactory;
import net.recommenders.rival.core.DataModelIF;
import net.recommenders.rival.core.DataModelUtils;
import net.recommenders.rival.evaluation.metric.error.MAE;
import net.recommenders.rival.evaluation.metric.error.RMSE;
import net.recommenders.rival.evaluation.metric.ranking.NDCG;
import net.recommenders.rival.evaluation.metric.ranking.Precision;
import net.recommenders.rival.evaluation.metric.ranking.Recall;
import net.recommenders.rival.evaluation.strategy.EvaluationStrategy;
import net.recommenders.rival.evaluation.strategy.RelPlusN;
import net.recommenders.rival.evaluation.strategy.TestItems;
import net.recommenders.rival.evaluation.strategy.UserTest;
import net.recommenders.rival.split.splitter.CrossValidationSplitter;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
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
 * Created by fpena on 18/07/2017.
 */
public class RichContextEvaluator {

    private final int numFolds;
    private final int at;
    private final int additionalItems;
    private final int numTopics;
    private final double relevanceThreshold;
    private final long seed;
    private final Strategy strategy;
    private final ContextFormat contextFormat;
    private final Dataset dataset;
    private final String outputFile;
    private final boolean coldStart;


    private Map<String, Review> reviewsMap;
    private String ratingsFolderPath;
    private String jsonRatingsFile;
    private int numUsers;
    private int numItems;

    private static final String RATING = "rating";
    private static final String RANKING = "ranking";


    private enum EvaluationSet {
        TRAIN_USERS,
        TEST_USERS,
        TEST_ONLY_USERS,
        TRAIN_ONLY_USERS,
    }

    private static EvaluationSet evaluationSet;

    private enum Strategy {
        ALL_ITEMS(RATING),
        REL_PLUS_N(RANKING),
        TEST_ITEMS(RATING),
        TRAIN_ITEMS(RATING),
        USER_TEST(RATING);

        private final String predictionType;

        Strategy(String predictionType) {
            this.predictionType = predictionType;
        }

        public String getPredictionType() {
            return predictionType;
        }
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

    private enum ProcessingTask {
        PREPARE_LIBFM,
        PREPARE_CARSKIT,
        PROCESS_LIBFM_RESULTS,
        PROCESS_CARSKIT_RESULTS,
        EVALUATE_LIBFM_RESULTS
    }


    public RichContextEvaluator(
            String cacheFolder, String outputFolder, String propertiesFile)
            throws IOException {

        Properties properties = Properties.loadProperties(propertiesFile);
        numFolds = properties.getCrossValidationNumFolds();
        at = properties.getTopN();
        relevanceThreshold = properties.getRelevanceThreshold();
        seed = properties.getSeed();
        strategy = Strategy.valueOf(
                (properties.getStrategy().toUpperCase(Locale.ENGLISH)));
        additionalItems = properties.getTopnNumItems();
        contextFormat = ContextFormat.valueOf(
                properties.getContextFormat().toUpperCase(Locale.ENGLISH));
        dataset = RichContextEvaluator.Dataset.valueOf(
                properties.getDataset().toUpperCase(Locale.ENGLISH));
        numTopics = properties.getNumTopics();
        coldStart = properties.getEvaluateColdStart();
        outputFile = outputFolder +
                "rival_" + properties.getDataset() + "_results_folds_4.csv";

        jsonRatingsFile = cacheFolder + dataset.toString().toLowerCase() +
                "_recsys_formatted_context_records_ensemble_" +
                "numtopics-" + numTopics + "_iterations-100_passes-10_targetreview-specific_" +
                "normalized_contextformat-" + contextFormat.toString().toLowerCase()
                + "_lang-en_bow-NN_document_level-review_targettype-context_" +
                "min_item_reviews-10.json";


        init();
    }

    private void init() throws IOException {

        File jsonFile =  new File(jsonRatingsFile);
        String jsonFileName = jsonFile.getName();
        String jsonFileParentFolder = jsonFile.getParent();
        String rivalFolderPath = jsonFileParentFolder + "/rival/";

        File rivalDir = new File(rivalFolderPath);
        if (!rivalDir.exists()) {
            if (!rivalDir.mkdir()) {
                throw new IOException("Directory " + rivalDir + " could not be created");
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
                throw new IOException("Directory " + ratingsDir + " could not be created");
            }
        }

        System.out.println("File name: " + jsonFileName);
        System.out.println("Parent folder: " + jsonFileParentFolder);
        System.out.println("Ratings folder: " + ratingsFolderPath);
    }


    /**
     * Downloads a dataset and stores the splits generated from it.
     */
    private void prepareSplits() throws IOException {

        String dataFile = jsonRatingsFile;
        boolean perUser = false;
        JsonParser parser = new JsonParser();

        DataModelIF<Long, Long> data = null;
        data = parser.parseData(new File(dataFile));

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
                        numFolds, perUser, seed).split(data);
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

            // If we want to test cold-start scenarios
            if (coldStart) {

                // Filter the users from the training and testing sets
                List<Long> allUsers = new ArrayList<>(usersSet);
                Collections.shuffle(allUsers, new Random(i));

                int cutPoint = allUsers.size() - allUsers.size() / numFolds;
                List<Long> trainUsers = allUsers.subList(0, cutPoint);
                List<Long> testUsers = allUsers.subList(cutPoint, allUsers.size());
                for (Long trainUser : trainUsers) {
                    test.getUserItemPreferences().remove(trainUser);
                }
                for (Long testUser : testUsers) {
                    training.getUserItemPreferences().remove(testUser);
                }
            }

            String trainingFile = foldPath + "train.csv";
            String testFile = foldPath + "test.csv";
            boolean overwrite = true;
            DataModelUtils.saveDataModel(training, trainingFile, overwrite, "\t");
            DataModelUtils.saveDataModel(test, testFile, overwrite, "\t");
        }
    }


    private void transformSplitToCarskit(int fold) throws IOException {

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


    private void transformSplitsToCarskit() throws IOException {

        for (int fold = 0; fold < numFolds; fold++) {
            transformSplitToCarskit(fold);
        }
    }


    private void transformSplitToLibfm(int fold) throws IOException {

        System.out.println("Transform split " + fold + " to LibFM");

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
//        String libfmTestFile = foldPath + "libfm_test.libfm";
        String libfmPredictionsFile = foldPath + "libfm_predictions_" +
                strategy.getPredictionType() + ".libfm";

        Map<String, Integer> oneHotIdMap =
                LibfmExporter.getOneHot(reviewsMap.values());

        LibfmExporter.exportRecommendations(
                completeTrainReviews, libfmTrainFile, oneHotIdMap);

        switch (strategy) {
            case TEST_ITEMS:
            case USER_TEST:
                LibfmExporter.exportRecommendations(
                        completeTestReviews, libfmPredictionsFile, oneHotIdMap);
                break;
            case REL_PLUS_N: {
                LibfmExporter.exportRankingPredictionsFile(
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


    private void transformSplitsToLibfm() throws IOException {

        for (int fold = 0; fold < numFolds; fold++) {
            transformSplitToLibfm(fold);
        }
    }


    private void parseRecommendationResults(String algorithm) throws IOException, InterruptedException {

        System.out.println("Parse Recommendation Results");

        for (int fold = 0; fold < numFolds; fold++) {

            // Collect the results from the recommender
            String recommendationsFile = getRecommendationsFileName(
                    ratingsFolderPath, algorithm, fold, at);

            List<Review> recommendations = (at < 1) ?
                    CarskitResultsParser.parseRatingResults(recommendationsFile) :
                    CarskitResultsParser.parseRankingResults(recommendationsFile);

            String rivalRecommendationsFile =
                    ratingsFolderPath + "fold_" + fold + "/recs_" + algorithm +
                            "_" + strategy.getPredictionType()  + ".csv";
            System.out.println("Recommendations file name: " + rivalRecommendationsFile);
            CarskitExporter.exportRecommendationsToCsv(
                    recommendations, rivalRecommendationsFile);
        }
    }


    private void parseRecommendationResultsLibfm() throws IOException, InterruptedException {

        System.out.println("Parse Recommendation Results LibFM");

        for (int fold = 0; fold < numFolds; fold++) {

            // Collect the results from the recommender
            String foldPath = ratingsFolderPath + "fold_" + fold + "/";
//            String testFile = foldPath + "test.csv";
            String predictionsFile;
            String libfmResultsFile = foldPath + "libfm_results_" +
                    strategy.getPredictionType() + ".txt";
            String rivalRecommendationsFile =
                    ratingsFolderPath + "fold_" + fold + "/recs_libfm_" +
                            strategy.getPredictionType()  + ".csv";

            switch (strategy) {
                case TEST_ITEMS:
                case USER_TEST:
                    predictionsFile = foldPath + "test.csv";
                    LibfmResultsParser.parseResults(
                            predictionsFile, libfmResultsFile, false, rivalRecommendationsFile);
                    break;
                case REL_PLUS_N:
                    predictionsFile = foldPath + "predictions.csv";
                    LibfmResultsParser.parseResults(
                            predictionsFile, libfmResultsFile, true, rivalRecommendationsFile);
                    break;
                default:
                    String msg = strategy.toString() +
                            " evaluation strategy not supported";
                    throw new UnsupportedOperationException(msg);
            }

            System.out.println("Recommendations file name: " + rivalRecommendationsFile);
        }
    }


    private String getRecommendationsFileName(
            String workingPath, String algorithm, int foldIndex, int topN) {

        String filePath;
        String carskitWorkingPath =
                workingPath + "fold_" + foldIndex + "/CARSKit.Workspace/";

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


    private void writeResultsToFile(
            List<Map<String, String>> resultsList) throws IOException {

        String[] headers = {
                "Algorithm",
                "Num_Topics",
                "Strategy",
                "Context_Format",
                "Cold-start",
                "NDCG@" + at,
                "Precision@" + at,
                "Recall@" + at,
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

        File resultsFile = new File(outputFile);
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


    private void postProcess(String algorithm)
            throws IOException, InterruptedException {

        List<Map<String, String>> resultsList = new ArrayList<>();

        parseRecommendationResults(algorithm);
        prepareStrategy(algorithm);
        resultsList.add(evaluate(algorithm));
        writeResultsToFile(resultsList);
    }


    private Map<String, String> evaluate(String algorithm) throws IOException {

        System.out.println("Evaluate");

        double ndcgRes = 0.0;
        double recallRes = 0.0;
        double precisionRes = 0.0;
        double rmseRes = 0.0;
        double maeRes = 0.0;
        Map<String, String> results =  new HashMap<>();
        for (int i = 0; i < numFolds; i++) {
            String foldPath = ratingsFolderPath + "fold_" + i + "/";
            File testFile = new File(foldPath + "test.csv");
            File strategyFile = new File(foldPath + "strategymodel_" + algorithm + "_" + strategy.toString().toLowerCase() + ".csv");
            DataModelIF<Long, Long> testModel = new CsvParser().parseData(testFile);
            DataModelIF<Long, Long> recModel;

            switch (strategy) {
                case TEST_ITEMS:
                case USER_TEST:
                    recModel = new CsvParser().parseData(strategyFile);
                    break;
                case REL_PLUS_N:
                    File trainingFile = new File(foldPath + "train.csv");
                    DataModelIF<Long, Long> trainModel = new CsvParser().parseData(trainingFile);
                    Map<Long, Integer> trainMap = countUserReviewFrequency(trainModel);
                    Set<Long> trainUsers = trainMap.keySet();
                    System.out.println("Num train users = " + trainUsers.size());
                    Map<Long, Integer> testMap = countUserReviewFrequency(testModel);
                    Set<Long> testUsers = testMap.keySet();
                    System.out.println("Num test users = " + testUsers.size());
                    Set<Long> users;

                    System.out.println("Evaluation set: " + evaluationSet);

                    switch (evaluationSet) {

                        case TEST_USERS:
                            users = testUsers;
                            break;
                        case TRAIN_USERS:
                            users = trainUsers;
                            break;
                        case TEST_ONLY_USERS:
                            testUsers.removeAll(trainUsers);
                            users = testUsers;
                            break;
                        case TRAIN_ONLY_USERS:
                            trainUsers.removeAll(testUsers);
                            users = trainUsers;
                            break;
                        default:
                            String msg =
                                    "Evaluation set " + evaluationSet +
                                    " not supported";
                            throw new UnsupportedOperationException(msg);

                    }

                    recModel = new CsvParser().parseData(strategyFile, users);
//                    recModel = new CsvParser().parseData(strategyFile);
                    break;
                default:
                    throw new UnsupportedOperationException(
                            strategy.toString() + " evaluation Strategy not supported");
            }


            NDCG<Long, Long> ndcg = new NDCG<>(recModel, testModel, new int[]{at});
            ndcg.compute();
            ndcgRes += ndcg.getValueAt(at);
            results.put("fold_" + i + "_ndcg", String.valueOf(ndcg.getValueAt(at)));

            Recall<Long, Long> recall = new Recall<>(recModel, testModel, relevanceThreshold, new int[]{at});
            recall.compute();
            recallRes += recall.getValueAt(at);
            results.put("fold_" + i + "_recall", String.valueOf(recall.getValueAt(at)));

            RMSE<Long, Long> rmse = new RMSE<>(recModel, testModel);
            rmse.compute();
            rmseRes += rmse.getValue();
            results.put("fold_" + i + "_rmse", String.valueOf(rmse.getValue()));

            MAE<Long, Long> mae = new MAE<>(recModel, testModel);
            mae.compute();
            maeRes += mae.getValue();
            results.put("fold_" + i + "_mae", String.valueOf(mae.getValue()));

            Precision<Long, Long> precision = new Precision<>(recModel, testModel, relevanceThreshold, new int[]{at});
            precision.compute();
            precisionRes += precision.getValueAt(at);
            results.put("fold_" + i + "_precision", String.valueOf(precision.getValueAt(at)));
        }

        results.put("Dataset", dataset.toString());
        results.put("Algorithm", algorithm);
        results.put("Num_Topics", String.valueOf(numTopics));
        results.put("Strategy", strategy.toString());
        results.put("Context_Format", contextFormat.toString());
        results.put("Cold-start", String.valueOf(coldStart));
        results.put("NDCG@" + at, String.valueOf(ndcgRes / numFolds));
        results.put("Precision@" + at, String.valueOf(precisionRes / numFolds));
        results.put("Recall@" + at, String.valueOf(recallRes / numFolds));
        results.put("RMSE", String.valueOf(rmseRes / numFolds));
        results.put("MAE", String.valueOf(maeRes / numFolds));

        System.out.println("Dataset: " + dataset.toString());
        System.out.println("Algorithm: " + algorithm);
        System.out.println("Num Topics: " + numTopics);
        System.out.println("Strategy: " + strategy.toString());
        System.out.println("Context_Format: " + contextFormat.toString());
        System.out.println("Cold-start: " + coldStart);
        System.out.println("NDCG@" + at + ": " + ndcgRes / numFolds);
        System.out.println("Precision@" + at + ": " + precisionRes / numFolds);
        System.out.println("Recall@" + at + ": " + recallRes / numFolds);
        System.out.println("RMSE: " + rmseRes / numFolds);
        System.out.println("MAE: " + maeRes / numFolds);
//        System.out.println("P@" + AT + ": " + precisionRes / nFolds);

        return results;
    }


    private static Set<Long> getElementsByFrequency(
            Map<Long, Integer> frequencyMap, int min, int max) {

        Set<Long> elementsSet = new HashSet<>();

        for (Map.Entry<Long, Integer> entry : frequencyMap.entrySet()) {
            if (max >= entry.getValue() && min <= entry.getValue()) {
                elementsSet.add(entry.getKey());
            }
        }

//        System.out.println(elementsSet);

        return elementsSet;
    }


    private static Map<Long, Integer> countUserReviewFrequency(
            DataModelIF<Long, Long> dataModel) {

        Map<Long, Integer> frequencyMap = new HashMap<>();
        Map<Long, Map<Long, Double>> userItemPreferences =
                dataModel.getUserItemPreferences();
        for (Map.Entry<Long, Map<Long, Double>> entry : userItemPreferences.entrySet()) {
            frequencyMap.put(entry.getKey(), entry.getValue().size());
        }

//        System.out.println("User review frequency: " + frequencyMap);

        return frequencyMap;
    }


    private static void prepareLibfm(
            String cacheFolder, String outputFolder, String propertiesFile)
            throws IOException {

        RichContextEvaluator evaluator = new RichContextEvaluator(
                cacheFolder, outputFolder, propertiesFile);
        evaluator.prepareSplits();
        evaluator.transformSplitsToLibfm();
    }


    private static void prepareCarskit(
            String jsonFile, String outputFolder, String propertiesFile)
            throws IOException {


        RichContextEvaluator evaluator = new RichContextEvaluator(
                jsonFile, outputFolder, propertiesFile);
        evaluator.prepareSplits();
        evaluator.transformSplitsToCarskit();
    }


    private static void processLibfmResults(
            String jsonFile, String outputFolder, String propertiesFile)
            throws IOException, InterruptedException {

        RichContextEvaluator evaluator = new RichContextEvaluator(
                jsonFile, outputFolder, propertiesFile);
        evaluator.prepareSplits();
        evaluator.parseRecommendationResultsLibfm();
        evaluator.prepareStrategy("libfm");
        Map<String, String> results = evaluator.evaluate("libfm");

        List<Map<String, String>> resultsList = new ArrayList<>();
        resultsList.add(results);
        evaluator.writeResultsToFile(resultsList);
    }


    private static void prepareStrategyAndEvaluate(
            String jsonFile, String outputFolder, String propertiesFile)
            throws IOException, InterruptedException {

        RichContextEvaluator evaluator = new RichContextEvaluator(
                jsonFile, outputFolder, propertiesFile);
        evaluator.prepareStrategy("libfm");
        Map<String, String> results = evaluator.evaluate("libfm");

        List<Map<String, String>> resultsList = new ArrayList<>();
        resultsList.add(results);
        evaluator.writeResultsToFile(resultsList);
    }


    private static void processCarskitResults(
            String jsonFile, String outputFolder, String propertiesFile)
            throws IOException, InterruptedException {

        RichContextEvaluator evaluator = new RichContextEvaluator(
                jsonFile, outputFolder, propertiesFile);
        evaluator.prepareSplits();

        String[] algorithms = {
                "GlobalAvg",
                "UserAvg",
                "ItemAvg",
                "UserItemAvg",
                "SlopeOne",
                "PMF",
                "BPMF",
                "BiasedMF",
                "NMF",
                "CAMF_CI", "CAMF_CU",
                "CAMF_CUCI",
                "SLIM",
                "BPR",
                "LRMF",
                "CSLIM_C", "CSLIM_CI",
                "CSLIM_CU",
        };

        int progress = 1;
        for (String algorithm : algorithms) {

            System.out.println(
                    "\n\nProgress: " + progress + "/" + algorithms.length);
            evaluator.postProcess(algorithm);
            progress++;
        }
    }


    private void prepareRankingStrategy(String algorithm) throws IOException {

        System.out.println("Prepare Ranking Strategy");

        for (int i = 0; i < numFolds; i++) {

            String foldPath = ratingsFolderPath + "fold_" + i + "/";
            File trainingFile = new File(foldPath + "train.csv");
            File testFile = new File(foldPath + "test.csv");
            File recFile = new File(foldPath + "recs_" + algorithm + "_" +
                    strategy.getPredictionType()  + ".csv");
            DataModelIF<Long, Long> trainingModel;
            DataModelIF<Long, Long> testModel;
            DataModelIF<Long, Long> recModel;
            System.out.println("Parsing Training Model");
            trainingModel = new CsvParser().parseData(trainingFile);
            System.out.println("Parsing Test Model");
            testModel = new CsvParser().parseData(testFile);
            System.out.println("Parsing Recommendation Model");
            recModel = new CsvParser().parseData(recFile);

            System.out.println("Recommendation model num users: " + recModel.getNumUsers());
            System.out.println("Recommendation model num items: " + recModel.getNumItems());
            System.out.println("Recommendation model num predictions: " + recModel.getUserItemPreferences().size());

            EvaluationStrategy<Long, Long> evaluationStrategy =
                    new RelPlusN(trainingModel, testModel, additionalItems, relevanceThreshold, seed);

            DataModelIF<Long, Long> modelToEval = DataModelFactory.getDefaultModel();
            for (Long user : recModel.getUsers()) {

//                System.out.println("Candidate items to rank: " + evaluationStrategy.getCandidateItemsToRank(user).size());

                Map<Long, Double> itemPreferences = recModel.getUserItemPreferences().get(user);

                for (Long item : evaluationStrategy.getCandidateItemsToRank(user)) {

                    if (itemPreferences.containsKey(item)) {
                        modelToEval.addPreference(user, item, itemPreferences.get(item));
                    }
                }
            }
            DataModelUtils.saveDataModel(modelToEval, foldPath + "strategymodel_" + algorithm + "_" + strategy.toString().toLowerCase() + ".csv", true, "\t");
        }
    }


    private void prepareRatingStrategy(String algorithm) throws IOException {

        System.out.println("Prepare Rating Strategy");

        for (int i = 0; i < numFolds; i++) {

            String foldPath = ratingsFolderPath + "fold_" + i + "/";
            File trainingFile = new File(foldPath + "train.csv");
            File testFile = new File(foldPath + "test.csv");
            File recFile = new File(foldPath + "recs_" + algorithm + "_" +
                    strategy.getPredictionType()  + ".csv");
            DataModelIF<Long, Long> trainingModel;
            DataModelIF<Long, Long> testModel;
            DataModelIF<Long, Long> recModel;
            trainingModel = new CsvParser().parseData(trainingFile);
            testModel = new CsvParser().parseData(testFile);
            recModel = new CsvParser().parseData(recFile);

            EvaluationStrategy<Long, Long> evaluationStrategy;

            switch (strategy) {
                case TEST_ITEMS:
                    evaluationStrategy = new TestItems((DataModel)trainingModel, (DataModel)testModel, relevanceThreshold);
                    break;
                case USER_TEST:
                    evaluationStrategy = new UserTest((DataModel)trainingModel, (DataModel)testModel, relevanceThreshold);
                    break;
                default:
                    String msg = strategy.toString() +
                            " evaluation strategy not supported for rating";
                    throw new UnsupportedOperationException(msg);
            }


            DataModelIF<Long, Long> modelToEval = DataModelFactory.getDefaultModel();
            for (Long user : recModel.getUsers()) {

//                System.out.println("Candidate items to rank: " + evaluationStrategy.getCandidateItemsToRank(user).size());

                for (Long item : evaluationStrategy.getCandidateItemsToRank(user)) {
                    if (recModel.getUserItemPreferences().get(user).containsKey(item)) {
                        modelToEval.addPreference(user, item, recModel.getUserItemPreferences().get(user).get(item));
                    }
                }
            }
            DataModelUtils.saveDataModel(
                    modelToEval,
                    foldPath + "strategymodel_" + algorithm + "_" + strategy.toString().toLowerCase() + ".csv",
                    true, "\t");
        }
    }


    private void prepareStrategy(String algorithm) throws IOException {

        switch (strategy) {
            case TEST_ITEMS:
            case USER_TEST:
                prepareRatingStrategy(algorithm);
                break;
            case REL_PLUS_N:
                prepareRankingStrategy(algorithm);
                break;
            default:
                String msg = strategy.toString() +
                        " evaluation strategy not supported";
                throw new UnsupportedOperationException(msg);
        }
    }


    public static void main(final String[] args) throws IOException, InterruptedException, ParseException {

        // create Options object
        Options options = new Options();

        // add t option
        options.addOption("t", true, "The processing task");
        options.addOption("d", true, "The folder containing the data file");
        options.addOption("o", true, "The folder containing the output file");
        options.addOption("p", true, "The properties file path");
        options.addOption("s", true, "The evaluation set");

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);

        String defaultOutputFolder = "/Users/fpena/tmp/";
        String defaultPropertiesFile =
                "/Users/fpena/UCC/Thesis/projects/yelp/source/python/" +
                        "properties.yaml";
        String defaultCacheFolder =
                "/Users/fpena/UCC/Thesis/datasets/context/stuff/cache_context/";
        String defaultEvaluationSet = "TRAIN_USERS";

        ProcessingTask processingTask;
        processingTask =
                ProcessingTask.valueOf(cmd.getOptionValue("t").toUpperCase());
        String cacheFolder = cmd.getOptionValue("d", defaultCacheFolder);
        String outputFolder = cmd.getOptionValue("o", defaultOutputFolder);
        String propertiesFile = cmd.getOptionValue("p", defaultPropertiesFile);
        evaluationSet = EvaluationSet.valueOf(
                cmd.getOptionValue("s", defaultEvaluationSet).toUpperCase());

        long startTime = System.currentTimeMillis();

        switch (processingTask) {
            case PREPARE_LIBFM:
                prepareLibfm(cacheFolder, outputFolder, propertiesFile);
                break;
            case PREPARE_CARSKIT:
                prepareCarskit(cacheFolder, outputFolder, propertiesFile);
                break;
            case PROCESS_LIBFM_RESULTS:
                processLibfmResults(cacheFolder, outputFolder, propertiesFile);
                break;
            case PROCESS_CARSKIT_RESULTS:
                processCarskitResults(cacheFolder, outputFolder, propertiesFile);
                break;
            case EVALUATE_LIBFM_RESULTS:
                prepareStrategyAndEvaluate(cacheFolder, outputFolder, propertiesFile);
                break;
            default:
                throw new UnsupportedOperationException(
                        "Unknown processing task: " + processingTask.toString());
        }

        long endTime   = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Running time: " + (totalTime/1000));
    }
}
