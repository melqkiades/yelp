package org.insightcentre.richcontext;

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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Created by fpena on 05/09/2017.
 */
public class RichContextResultsProcessor {


    private int numFolds;
    private int at;
    private int additionalItems;
    private int numTopics;
    private double relevanceThreshold;
    private long seed;
    private Strategy strategy;
    private ContextFormat contextFormat;
    private Dataset dataset;
    private String outputFile;
    private boolean coldStart;


    private String ratingsFolderPath;

    private final String[] headers;

    private static final String RATING = "rating";
    private static final String RANKING = "ranking";

    public enum EvaluationSet {
        TRAIN_USERS,
        TEST_USERS,
        TEST_ONLY_USERS,
        TRAIN_ONLY_USERS,
    }

    private EvaluationSet evaluationSet;

    public enum Strategy {
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

    public enum ContextFormat {
        NO_CONTEXT,
        CONTEXT_TOPIC_WEIGHTS,
        TOP_WORDS,
        PREDEFINED_CONTEXT,
        TOPIC_PREDEFINED_CONTEXT
    }

    public enum Dataset {
        YELP_HOTEL,
        YELP_RESTAURANT,
        FOURCITY_HOTEL,
    }

    public enum ProcessingTask {
        PREPARE_LIBFM,
        PROCESS_LIBFM_RESULTS,
        EVALUATE_LIBFM_RESULTS
    }


    public RichContextResultsProcessor(
            String cacheFolder, String outputFolder, String propertiesFile,
            EvaluationSet evaluationSet, Integer paramNumTopics)
            throws IOException {


        Properties properties = Properties.loadProperties(propertiesFile);
        numFolds = properties.getCrossValidationNumFolds();
        at = properties.getTopN();
        relevanceThreshold = properties.getRelevanceThreshold();
        seed = properties.getSeed();
        strategy = Strategy.valueOf(
                (properties.getStrategy().toUpperCase(Locale.ENGLISH)));
        additionalItems = properties.getTopnNumItems();
        contextFormat = RichContextResultsProcessor.ContextFormat.valueOf(
                properties.getContextFormat().toUpperCase(Locale.ENGLISH));
        dataset = RichContextResultsProcessor.Dataset.valueOf(
                properties.getDataset().toUpperCase(Locale.ENGLISH));
        numTopics = (paramNumTopics == null) ?
                properties.getNumTopics() :
                paramNumTopics;
        coldStart = properties.getEvaluateColdStart();
        outputFile = outputFolder +
                "rival_" + properties.getDataset() + "_results_folds_4.csv";

        String jsonRatingsFile = cacheFolder + dataset.toString().toLowerCase() +
                "_recsys_formatted_context_records_ensemble_" +
                "numtopics-" + numTopics + "_iterations-100_passes-10_targetreview-specific_" +
                "normalized_contextformat-" + contextFormat.toString().toLowerCase()
                + "_lang-en_bow-NN_document_level-review_targettype-context_" +
                "min_item_reviews-10.json";
        this.evaluationSet = evaluationSet;

        this.ratingsFolderPath = Utils.getRatingsFolderPath(jsonRatingsFile);

        headers = new String[] {
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
    }


    public String getOutputFile() {
        return outputFile;
    }

    public String[] getHeaders() {
        return headers;
    }


    /**
     * Takes the file generated by the recommender which stores the predictions
     * and parses the results, generating another file that is RiVal compatible
     */
    private void parseRecommendationResultsLibfm() throws IOException, InterruptedException {

        System.err.println("Parse Recommendation Results LibFM");

        for (int fold = 0; fold < numFolds; fold++) {

            // Collect the results from the recommender
            String foldPath = ratingsFolderPath + "fold_" + fold + "/";
//            String testFile = foldPath + "test.csv";
            String predictionsFile;
            String libfmResultsFile = foldPath + "libfm_results_" +
                    strategy.getPredictionType() + ".txt";

            // Results will be stored in this file, which is RiVal compatible
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

            System.err.println("Recommendations file name: " + rivalRecommendationsFile);
        }
    }


    /**
     * Prepares the strategy file to be used by Rival
     *
     * @param algorithm the algorithm used to make the predictions
     */
    private void prepareStrategy(String algorithm) throws IOException {

        System.err.println("Prepare Rating Strategy");

        for (int i = 0; i < numFolds; i++) {

            String foldPath = ratingsFolderPath + "fold_" + i + "/";
            File trainingFile = new File(foldPath + "train.csv");
            File testFile = new File(foldPath + "test.csv");
            File recFile = new File(foldPath + "recs_" + algorithm + "_" +
                    strategy.getPredictionType()  + ".csv");
            DataModelIF<Long, Long> trainingModel;
            DataModelIF<Long, Long> testModel;
            DataModelIF<Long, Long> recModel;
            System.err.println("Parsing Training Model");
            trainingModel = new CsvParser().parseData(trainingFile);
            System.err.println("Parsing Test Model");
            testModel = new CsvParser().parseData(testFile);
            System.err.println("Parsing Recommendation Model");
            recModel = new CsvParser().parseData(recFile);

            EvaluationStrategy<Long, Long> evaluationStrategy;

            switch (strategy) {
                case TEST_ITEMS:
                    evaluationStrategy = new TestItems(
                            (DataModel)trainingModel, (DataModel)testModel,
                            relevanceThreshold);
                    break;
                case USER_TEST:
                    evaluationStrategy = new UserTest(
                            (DataModel)trainingModel, (DataModel)testModel,
                            relevanceThreshold);
                    break;
                case REL_PLUS_N:
                    evaluationStrategy = new RelPlusN(
                            trainingModel, testModel, additionalItems,
                            relevanceThreshold, seed);
                    break;
                default:
                    String msg = strategy.toString() +
                            " evaluation strategy not supported for rating";
                    throw new UnsupportedOperationException(msg);
            }


            DataModelIF<Long, Long> modelToEval = DataModelFactory.getDefaultModel();
            for (Long user : recModel.getUsers()) {
                for (Long item : evaluationStrategy.getCandidateItemsToRank(user)) {
                    if (!Double.isNaN(recModel.getUserItemPreference(user, item))) {
                        modelToEval.addPreference(user, item, recModel.getUserItemPreference(user, item));
                    }
                }
            }
            DataModelUtils.saveDataModel(
                    modelToEval,
                    foldPath + "strategymodel_" + algorithm + "_" + strategy.toString().toLowerCase() + ".csv",
                    true, "\t");
        }
    }


    /**
     * Evaluates the performance of the recommender system indicated by the
     * {@code algorithm} parameter. This method requires that the strategy files
     * are already generated
     *
     * @param algorithm the algorithm used to make the predictions
     * @return a {@link Map} with the hyperparameters of the algorithm and the
     * performance metrics.
     */
    private Map<String, String> evaluate(String algorithm) throws IOException {

        System.err.println("Evaluate");

        double ndcgRes = 0.0;
        double recallRes = 0.0;
        double precisionRes = 0.0;
        double rmseRes = 0.0;
        double maeRes = 0.0;
        Map<String, String> results =  new HashMap<>();
        for (int i = 0; i < numFolds; i++) {
            String foldPath = ratingsFolderPath + "fold_" + i + "/";
            File testFile = new File(foldPath + "test.csv");
            String strategyFileName = foldPath + "strategymodel_" + algorithm +
                    "_" + strategy.toString().toLowerCase() + ".csv";
            File strategyFile = new File(strategyFileName);
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
                    Set<Long> trainUsers = new HashSet<>();
                    for (Long user : trainModel.getUsers()) {
                        trainUsers.add(user);
                    }
                    System.err.println("Num train users = " + trainUsers.size());

                    Set<Long> testUsers = new HashSet<>();
                    for (Long user : testModel.getUsers()) {
                        testUsers.add(user);
                    }
                    System.err.println("Num test users = " + testUsers.size());
                    Set<Long> users;

                    System.err.println("Evaluation set: " + evaluationSet);

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

            Recall<Long, Long> recall = new Recall<>(
                    recModel, testModel, relevanceThreshold, new int[]{at});
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

            Precision<Long, Long> precision = new Precision<>(
                    recModel, testModel, relevanceThreshold, new int[]{at});
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

        System.err.println("Dataset: " + dataset.toString());
        System.err.println("Algorithm: " + algorithm);
        System.err.println("Num Topics: " + numTopics);
        System.err.println("Strategy: " + strategy.toString());
        System.err.println("Context_Format: " + contextFormat.toString());
        System.err.println("Cold-start: " + coldStart);
        System.err.println("NDCG@" + at + ": " + ndcgRes / numFolds);
        System.err.println("Precision@" + at + ": " + precisionRes / numFolds);
        System.err.println("Recall@" + at + ": " + recallRes / numFolds);
        System.err.println("RMSE: " + rmseRes / numFolds);
        System.err.println("MAE: " + maeRes / numFolds);

        return results;
    }


    /**
     * Runs the whole evaluation cycle starting after the recommendations have
     * been done.
     *
     * @param cacheFolder the folder that contains the dataset to be evaluated
     * @param outputFolder the folder where the results are going to be exported
     * @param propertiesFile the file that contains the hyperparameters for the
     * recommender
     * @param evaluationSet an enum that indicates which users are going to be
     * in the evaluation set. The possible values are:
     * {@code TRAIN_USERS, TEST_USERS, TRAIN_ONLY_USERS, TEST_ONLY_USERS}
     */
    public static void processLibfmResults(
            String cacheFolder, String outputFolder, String propertiesFile,
            EvaluationSet evaluationSet, Integer numTopics)
            throws IOException, InterruptedException {

        RichContextResultsProcessor evaluator = new RichContextResultsProcessor(
                cacheFolder, outputFolder, propertiesFile, evaluationSet,
                numTopics);
        evaluator.parseRecommendationResultsLibfm();
        evaluator.prepareStrategy("libfm");
        Map<String, String> results = evaluator.evaluate("libfm");

        List<Map<String, String>> resultsList = new ArrayList<>();
        resultsList.add(results);
        Utils.writeResultsToFile(
                resultsList, evaluator.getOutputFile(), evaluator.getHeaders());
    }
}
