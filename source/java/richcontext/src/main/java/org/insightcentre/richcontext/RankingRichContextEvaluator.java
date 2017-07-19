package org.insightcentre.richcontext;

import net.recommenders.rival.core.DataModelIF;
import net.recommenders.rival.evaluation.strategy.EvaluationStrategy;
import net.recommenders.rival.evaluation.strategy.RelPlusN;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.util.Map;

/**
 * Created by fpena on 23/03/2017.
 */
public class RankingRichContextEvaluator extends RichContextEvaluator {


    public RankingRichContextEvaluator(
            String cacheFolder, String outputFolder, String propertiesFile)
            throws IOException {

        super(cacheFolder, outputFolder, propertiesFile);
    }

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

        ProcessingTask processingTask =
                ProcessingTask.valueOf(cmd.getOptionValue("t").toUpperCase());
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
}
