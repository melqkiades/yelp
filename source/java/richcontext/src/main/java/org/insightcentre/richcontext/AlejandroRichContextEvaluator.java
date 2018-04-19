package org.insightcentre.richcontext;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import java.io.IOException;

/**
 * Created by fpena on 05/09/2017.
 */
public class AlejandroRichContextEvaluator {


    public static void main(final String[] args) throws IOException, InterruptedException, ParseException {

        System.out.println("AlejandroRichContextEvaluator");

        // create Options object
        Options options = new Options();

        // add t option
        options.addOption("t", true, "The processing task");
        options.addOption("d", true, "The folder containing the data file");
        options.addOption("o", true, "The folder containing the output file");
        options.addOption("p", true, "The properties file path");
        options.addOption("s", true, "The evaluation set");
        options.addOption("k", true, "The number of topics");
        options.addOption("i", true, "The type of items");

        CommandLineParser parser = new DefaultParser();
        CommandLine cmd = parser.parse(options, args);

        String defaultOutputFolder = "/Users/fpena/tmp/";
        String defaultPropertiesFile =
                "/Users/fpena/UCC/Thesis/projects/yelp/source/python/" +
                        "properties.yaml";
        String defaultCacheFolder =
                "/Users/fpena/UCC/Thesis/datasets/context/stuff/cache_context/";
        String defaultEvaluationSet = "test_users";
//        String defaultProcessingTask = "prepare_libfm";
        String defaultProcessingTask = "process_libfm_results";

        RichContextResultsProcessor.ProcessingTask processingTask;
        processingTask =
                RichContextResultsProcessor.ProcessingTask.valueOf(cmd.getOptionValue("t", defaultProcessingTask).toUpperCase());
        String cacheFolder = cmd.getOptionValue("d", defaultCacheFolder);
        String outputFolder = cmd.getOptionValue("o", defaultOutputFolder);
        String propertiesFile = cmd.getOptionValue("p", defaultPropertiesFile);
        Integer numTopics = cmd.hasOption("k") ?
                Integer.parseInt(cmd.getOptionValue("k", defaultPropertiesFile)) :
                null;
        String itemType = cmd.hasOption("i") ?
                cmd.getOptionValue("i", defaultPropertiesFile) :
                null;
        RichContextResultsProcessor.EvaluationSet evaluationSet =
                RichContextResultsProcessor.EvaluationSet.valueOf(
                cmd.getOptionValue("s", defaultEvaluationSet).toUpperCase());

        long startTime = System.currentTimeMillis();

        switch (processingTask) {
            case PREPARE_LIBFM:
                RichContextDataInitializer.prepareLibfm(
                        cacheFolder, propertiesFile, numTopics, itemType);
                break;
            case PROCESS_LIBFM_RESULTS:
                RichContextResultsProcessor.processLibfmResults(
                        cacheFolder, outputFolder, propertiesFile,
                        evaluationSet, numTopics, itemType);
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
