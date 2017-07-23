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

import static org.insightcentre.richcontext.RichContextEvaluator.Strategy.*;

/**
 * Created by fpena on 18/07/2017.
 */
public class RichContextEvaluator {

    protected final int numFolds;
    protected final int at;
    protected final int additionalItems;
    protected final double relevanceThreshold;
    protected final long seed;
    protected final Strategy strategy;
    protected final ContextFormat contextFormat;
    protected final Dataset dataset;
    protected final String outputFile;


    protected Map<String, Review> reviewsMap;
    protected String ratingsFolderPath;
    protected String jsonRatingsFile;
    protected int numUsers;
    protected int numItems;


    protected enum Strategy {
        ALL_ITEMS,
        REL_PLUS_N,
        TEST_ITEMS,
        TRAIN_ITEMS,
        USER_TEST
    }

    protected enum ContextFormat {
        NO_CONTEXT,
        CONTEXT_TOPIC_WEIGHTS,
        TOP_WORDS,
        PREDEFINED_CONTEXT,
        TOPIC_PREDEFINED_CONTEXT
    }

    protected enum Dataset {
        YELP_HOTEL,
        YELP_RESTAURANT,
        FOURCITY_HOTEL,
    }

    protected enum ProcessingTask {
        PREPARE_LIBFM,
        PREPARE_CARSKIT,
        PROCESS_LIBFM_RESULTS,
        PROCESS_CARSKIT_RESULTS
    }


    public RichContextEvaluator(
            String cacheFolder, String outputFolder, String propertiesFile)
            throws IOException {

        Properties properties = Properties.loadProperties(propertiesFile);
        numFolds = properties.getCrossValidationNumFolds();
        at = properties.getTopN();
        relevanceThreshold = properties.getRelevanceThreshold();
        seed = properties.getSeed();
        strategy = valueOf(
                (properties.getStrategy().toUpperCase(Locale.ENGLISH)));
        additionalItems = properties.getTopnNumItems();
        contextFormat = ContextFormat.valueOf(
                properties.getContextFormat().toUpperCase(Locale.ENGLISH));
        dataset = RichContextEvaluator.Dataset.valueOf(
                properties.getDataset().toUpperCase(Locale.ENGLISH));
        int numTopics = properties.getNumTopics();
        outputFile = outputFolder +
                "rival_" + properties.getDataset() + "_results_folds_3.csv";

        jsonRatingsFile = cacheFolder + dataset.toString().toLowerCase() +
                "_recsys_formatted_context_records_ensemble_" +
                "numtopics-" + numTopics + "_iterations-100_passes-10_targetreview-specific_" +
                "normalized_contextformat-" + contextFormat.toString().toLowerCase()
                + "_lang-en_bow-NN_document_level-review_targettype-context_" +
                "min_item_reviews-10.json";


        init();
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
     */
    public void prepareSplits() {

        String dataFile = jsonRatingsFile;
        boolean perUser = false;
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


    public void transformSplitsToCarskit() throws IOException {

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
//        String libfmTestFile = foldPath + "libfm_test.libfm";
        String libfmPredictionsFile = foldPath + "libfm_predictions_" +
                strategy.toString().toLowerCase() + ".libfm";

        Map<String, Integer> oneHotIdMap =
                LibfmExporter.getOneHot(reviewsMap.values());

        LibfmExporter.exportRecommendations(
                completeTrainReviews, libfmTrainFile, oneHotIdMap);

        switch (strategy) {
            case TEST_ITEMS:
                LibfmExporter.exportRecommendations(
                        completeTestReviews, libfmPredictionsFile, oneHotIdMap);
                break;
            case REL_PLUS_N: {
                List<Review> reviewsForRanking =
                        LibfmExporter.getReviewsForRanking(
                                completeTrainReviews, completeTestReviews, oneHotIdMap);
                CarskitExporter.exportContextRecommendationsToCsv(
                        reviewsForRanking, predictionsFile);
                LibfmExporter.exportRankingPredictionsFile(
                        completeTrainReviews, completeTestReviews,
                        libfmPredictionsFile, oneHotIdMap
                );
                break;
            }
            default:
                throw new UnsupportedOperationException(
                        strategy.toString() + " evaluation Strategy not supported");
        }

    }


    public void transformSplitsToLibfm() throws IOException {

        for (int fold = 0; fold < numFolds; fold++) {
            transformSplitToLibfm(fold);
        }
    }


    public void parseRecommendationResults(String algorithm) throws IOException, InterruptedException {

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
                            "_" + strategy.toString().toLowerCase()  + ".csv";
            System.out.println("Recommendations file name: " + rivalRecommendationsFile);
            CarskitExporter.exportRecommendationsToCsv(
                    recommendations, rivalRecommendationsFile);
        }
    }


    public void parseRecommendationResultsLibfm() throws IOException, InterruptedException {

        System.out.println("Parse Recommendation Results LibFM");

        for (int fold = 0; fold < numFolds; fold++) {

            // Collect the results from the recommender
            String foldPath = ratingsFolderPath + "fold_" + fold + "/";
//            String testFile = foldPath + "test.csv";
            String predictionsFile;
            String libfmResultsFile = foldPath + "libfm_results_" +
                    strategy.toString().toLowerCase() + ".txt";
            List<Review> recommendations;

            switch (strategy) {
                case TEST_ITEMS:
                    predictionsFile = foldPath + "test.csv";
                    recommendations = LibfmResultsParser.parseRatingResults(
                            predictionsFile, libfmResultsFile, false);
                    break;
                case REL_PLUS_N:
                    predictionsFile = foldPath + "predictions.csv";
                    recommendations =
                            LibfmResultsParser.parseContextRatingResults(
                                    predictionsFile, libfmResultsFile);
                    break;
                default:
                    String msg = strategy.toString() +
                            " evaluation strategy not supported";
                    throw new UnsupportedOperationException(msg);
            }


            String rivalRecommendationsFile =
                    ratingsFolderPath + "fold_" + fold + "/recs_libfm_" +
                            strategy.toString().toLowerCase()  + ".csv";
            System.out.println("Recommendations file name: " + rivalRecommendationsFile);
            CarskitExporter.exportRecommendationsToCsv(
                    recommendations, rivalRecommendationsFile);
        }
    }


    public String getRecommendationsFileName(
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


    protected void writeResultsToFile(
            List<Map<String, String>> resultsList) throws IOException {

        String[] headers = {
                "Algorithm",
                "Strategy",
                "Context_Format",
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


    protected void postProcess(String algorithm)
            throws IOException, InterruptedException {

        List<Map<String, String>> resultsList = new ArrayList<>();

        parseRecommendationResults(algorithm);
        prepareStrategy(algorithm);
        resultsList.add(evaluate(algorithm));
        writeResultsToFile(resultsList);
    }


    protected Map<String, String> evaluate(String algorithm) throws IOException {

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
            File strategyFile = new File(foldPath + "strategymodel_" + algorithm + "_" + strategy.toString() + ".csv");
            DataModelIF<Long, Long> testModel = new CsvParser().parseData(testFile);
            DataModelIF<Long, Long> recModel;

            switch (strategy) {
                case TEST_ITEMS:
                    recModel = new CsvParser().parseData(strategyFile);
                    break;
                case REL_PLUS_N:
                    File trainingFile = new File(foldPath + "train.csv");
                    DataModelIF<Long, Long> trainModel = new CsvParser().parseData(trainingFile);
                    Map<Long, Integer> myMap = countUserReviewFrequency(trainModel);
                    Set<Long> users = getElementsByFrequency(myMap, 0, 10000000);
                    System.out.println("Num users in training set range: " + users.size());
                    recModel = new CsvParser().parseData(strategyFile, users);
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
            results.put("fold_" + i + "_recall", String.valueOf(ndcg.getValueAt(at)));

            RMSE<Long, Long> rmse = new RMSE<>(recModel, testModel);
            rmse.compute();
            rmseRes += rmse.getValue();
            results.put("fold_" + i + "_rmse", String.valueOf(ndcg.getValue()));

            MAE<Long, Long> mae = new MAE<>(recModel, testModel);
            mae.compute();
            maeRes += mae.getValue();
            results.put("fold_" + i + "_mae", String.valueOf(ndcg.getValue()));

            Precision<Long, Long> precision = new Precision<>(recModel, testModel, relevanceThreshold, new int[]{at});
            precision.compute();
            precisionRes += precision.getValueAt(at);
            results.put("fold_" + i + "_precision", String.valueOf(ndcg.getValueAt(at)));
        }

        results.put("Algorithm", algorithm);
        results.put("Strategy", strategy.toString());
        results.put("Context_Format", contextFormat.toString());
        results.put("NDCG@" + at, String.valueOf(ndcgRes / numFolds));
        results.put("Precision@" + at, String.valueOf(precisionRes / numFolds));
        results.put("Recall@" + at, String.valueOf(recallRes / numFolds));
        results.put("RMSE", String.valueOf(rmseRes / numFolds));
        results.put("MAE", String.valueOf(maeRes / numFolds));

        System.out.println("Algorithm: " + algorithm);
        System.out.println("Strategy: " + strategy.toString());
        System.out.println("Context_Format: " + contextFormat.toString());
        System.out.println("NDCG@" + at + ": " + ndcgRes / numFolds);
        System.out.println("Precision@" + at + ": " + precisionRes / numFolds);
        System.out.println("Recall@" + at + ": " + recallRes / numFolds);
        System.out.println("RMSE: " + rmseRes / numFolds);
        System.out.println("MAE: " + maeRes / numFolds);
//        System.out.println("P@" + AT + ": " + precisionRes / nFolds);

        return results;
    }


    protected static Set<Long> getElementsByFrequency(
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


    protected static Map<Long, Integer> countUserReviewFrequency(
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


    protected static void prepareLibfm(
            String cacheFolder, String outputFolder, String propertiesFile)
            throws IOException {

        RichContextEvaluator evaluator = new RichContextEvaluator(
                cacheFolder, outputFolder, propertiesFile);
        evaluator.prepareSplits();
        evaluator.transformSplitsToLibfm();
    }


    protected static void prepareCarskit(
            String jsonFile, String outputFolder, String propertiesFile)
            throws IOException {


        RichContextEvaluator evaluator = new RichContextEvaluator(
                jsonFile, outputFolder, propertiesFile);
        evaluator.prepareSplits();
        evaluator.transformSplitsToCarskit();
    }


    protected static void processLibfmResults(
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


    protected static void processCarskitResults(
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


    public void prepareRankingStrategy(String algorithm) {

        System.out.println("Prepare Ranking Strategy");

        for (int i = 0; i < numFolds; i++) {

            String foldPath = ratingsFolderPath + "fold_" + i + "/";
            File trainingFile = new File(foldPath + "train.csv");
            File testFile = new File(foldPath + "test.csv");
            File recFile = new File(foldPath + "recs_" + algorithm + "_" +
                    strategy.toString().toLowerCase()  + ".csv");
            DataModelIF<Long, Long> trainingModel;
            DataModelIF<Long, Long> testModel;
            NewContextDataModel<Long, Long> recModel;
//            DataModelIF<Long, Long> recModel;
            try {
                trainingModel = new CsvParser().parseData(trainingFile);
                testModel = new CsvParser().parseData(testFile);
                recModel = new ContextParser().parseData2(recFile, "\t");
            } catch (IOException e) {
                e.printStackTrace();
                return;
            }

            System.out.println("Recommendation model num users: " + recModel.getNumUsers());
            System.out.println("Recommendation model num items: " + recModel.getNumItems());
            System.out.println("Recommendation model num predictions: " + recModel.getUserContextItemPreferences().size());
//            System.out.println("Recommendation model num predictions: " + recModel.getUserItemPreferences().size());

            EvaluationStrategy<Long, Long> evaluationStrategy =
                    new RelPlusN(trainingModel, testModel, additionalItems, relevanceThreshold, seed);

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
                modelToEval.saveDataModel(foldPath + "strategymodel_" + algorithm + "_" + strategy.toString() + ".csv", true, "\t");
//                DataModelUtils.saveDataModel(modelToEval, foldPath + "strategymodel_" + algorithm + "_" + STRATEGY.toString() + ".csv", true, "\t");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
    }


    public void prepareRatingStrategy(String algorithm) {

        System.out.println("Prepare Rating Strategy");

        for (int i = 0; i < numFolds; i++) {

            String foldPath = ratingsFolderPath + "fold_" + i + "/";
            File trainingFile = new File(foldPath + "train.csv");
            File testFile = new File(foldPath + "test.csv");
            File recFile = new File(foldPath + "recs_" + algorithm + "_" +
                    strategy.toString().toLowerCase()  + ".csv");
            DataModelIF<Long, Long> trainingModel;
            DataModelIF<Long, Long> testModel;
            DataModelIF<Long, Long> recModel;
            try {
                trainingModel = new CsvParser().parseData(trainingFile);
                testModel = new CsvParser().parseData(testFile);
                recModel = new CsvParser().parseData(recFile);
            } catch (IOException e) {
                e.printStackTrace();
                return;
            }

            EvaluationStrategy<Long, Long> evaluationStrategy =
                    new TestItems((DataModel)trainingModel, (DataModel)testModel, relevanceThreshold);

            DataModelIF<Long, Long> modelToEval = DataModelFactory.getDefaultModel();
            for (Long user : recModel.getUsers()) {
                for (Long item : evaluationStrategy.getCandidateItemsToRank(user)) {
                    if (recModel.getUserItemPreferences().get(user).containsKey(item)) {
                        modelToEval.addPreference(user, item, recModel.getUserItemPreferences().get(user).get(item));
                    }
                }
            }
            try {
                DataModelUtils.saveDataModel(
                        modelToEval,
                        foldPath + "strategymodel_" + algorithm + "_" + strategy.toString().toLowerCase() + ".csv", true, "\t");
            } catch (FileNotFoundException | UnsupportedEncodingException e) {
                e.printStackTrace();
            }
        }
    }


    protected void prepareStrategy(String algorithm) {

        switch (strategy) {
            case TEST_ITEMS:
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
//    protected abstract Map<String, String> evaluate(String algorithm) throws IOException;


    public static void main(final String[] args) throws IOException, InterruptedException, ParseException {

        // create Options object
        Options options = new Options();

        // add t option
        options.addOption("t", true, "The processing task");
        options.addOption("d", true, "The folder containing the data file");
        options.addOption("o", true, "The folder containing the output file");
        options.addOption("p", true, "The properties file path");

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);

        String defaultOutputFolder = "/Users/fpena/tmp/";
        String defaultPropertiesFile =
                "/Users/fpena/UCC/Thesis/projects/yelp/source/python/" +
                        "properties.yaml";
        String defaultCacheFolder =
                "/Users/fpena/UCC/Thesis/datasets/context/stuff/cache_context/";

        ProcessingTask processingTask;
//        processingTask =
//                ProcessingTask.valueOf(cmd.getOptionValue("t").toUpperCase());
        String cacheFolder = cmd.getOptionValue("d", defaultCacheFolder);
        String outputFolder = cmd.getOptionValue("o", defaultOutputFolder);
        String propertiesFile = cmd.getOptionValue("p", defaultPropertiesFile);

//        processingTask = ProcessingTask.PREPARE_LIBFM;
        processingTask = ProcessingTask.PROCESS_LIBFM_RESULTS;

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
            default:
                throw new UnsupportedOperationException(
                        "Unknown processing task");
        }

//        prepareLibfm(jsonFile, outputFolder, propertiesFile);
//        processLibfmResults(jsonFile, outputFolder, propertiesFile);
//        prepareCarskit(jsonFile, outputFolder, propertiesFile);
//        processCarskitResults(jsonFile, outputFolder, propertiesFile);

        long endTime   = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Running time: " + (totalTime/1000));
    }
}
