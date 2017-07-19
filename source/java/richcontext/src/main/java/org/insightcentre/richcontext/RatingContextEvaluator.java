package org.insightcentre.richcontext;

import net.recommenders.rival.core.DataModel;
import net.recommenders.rival.core.DataModelFactory;
import net.recommenders.rival.core.DataModelIF;
import net.recommenders.rival.core.DataModelUtils;
import net.recommenders.rival.evaluation.strategy.EvaluationStrategy;
import net.recommenders.rival.evaluation.strategy.TestItems;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;

/**
 * Created by fpena on 23/03/2017.
 */
public class RatingContextEvaluator extends RichContextEvaluator {



    public RatingContextEvaluator(
            String cacheFolder, String outputFolder, String propertiesFile)
            throws IOException {
        super(cacheFolder, outputFolder, propertiesFile);
    }

    public static void main(final String[] args) throws IOException, InterruptedException {

        long startTime = System.currentTimeMillis();

        String defaultOutputFolder = "/Users/fpena/tmp/";
        String defaultPropertiesFile =
                "/Users/fpena/UCC/Thesis/projects/yelp/source/python/" +
                        "properties.yaml";
        String defaultCacheFolder =
                "/Users/fpena/UCC/Thesis/datasets/context/stuff/cache_context/";

        RatingContextEvaluator evaluator = new RatingContextEvaluator(
                defaultCacheFolder, defaultOutputFolder, defaultPropertiesFile);
//        evaluator.prepareSplits();
//        evaluator.transformSplitsToCarskit(NUM_FOLDS);
//        evaluator.transformSplitsToLibfm();
        evaluator.parseRecommendationResultsLibfm();
        evaluator.prepareStrategy("libfm");
        evaluator.evaluate("libfm");

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
//                "SLIM",
//                "BPR",
//                "LRMF",
//                "CSLIM_C", "CSLIM_CI",
//                "CSLIM_CU",
        };

//        for (String algorithm : algorithms) {
//            evaluator.postProcess(NUM_FOLDS, algorithm);
//        }


//        RatingContextEvaluator evaluator = new RatingContextEvaluator("GlobalAvg", workingPath);

//        String fileName = getRecommendationsFileName(workingPath, "GlobalAvg", 1, -10);
//        System.out.println(fileName);

        long endTime   = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Running time: " + (totalTime/1000));
    }

/*
    @Override
    public void prepareStrategy(String algorithm) {

        System.out.println("Prepare Strategy");

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
    */
}
