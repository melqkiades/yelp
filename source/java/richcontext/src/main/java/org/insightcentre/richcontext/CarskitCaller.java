package org.insightcentre.richcontext;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.InterruptedIOException;
import java.util.Arrays;

/**
 * Created by fpena on 27/03/2017.
 */
public class CarskitCaller {

    private static final String JAVA_COMMAND = "java";
    private static final String CARSKIT_JAR = "CARSKit-v0.3.0.jar";
    private static final String CARSKIT_FOLDER = "/Users/fpena/tmp/CARSKit/";
    private static final String OUTPUT_FOLDER =
            "/Users/fpena/UCC/Thesis/datasets/context/stuff/carskit_results/";

    // This one has to be given as a parameter. If the folder is static then
    // you can't run experiments in parallel


    private CarskitCaller() {

    }


    public static void run(String confFile) throws IOException, InterruptedException {

        String jar_file = CARSKIT_FOLDER + "jar/" + CARSKIT_JAR;
        String[] command = {
                JAVA_COMMAND,
                "-jar",
                jar_file,
                "-c",
                confFile
        };

        System.out.println(Arrays.asList(command));

        ProcessBuilder processBuilder = new ProcessBuilder(command);
        System.out.println("Output folder: " + OUTPUT_FOLDER);
        processBuilder.directory(new File(OUTPUT_FOLDER));
        final Process process = processBuilder.start();

        //Read out dir output
        InputStream inputStream = process.getInputStream();
        InputStreamReader inputStreamReader = new InputStreamReader(inputStream);
        BufferedReader bufferedReader = new BufferedReader(inputStreamReader);

        InputStream errorStream = process.getErrorStream();
        InputStreamReader errorStreamReader = new InputStreamReader(errorStream);
        BufferedReader bufferedReaderError = new BufferedReader(errorStreamReader);

        System.out.printf("Output of running %s is:\n",
                Arrays.toString(command));
        String line;
        while ((line = bufferedReader.readLine()) != null) {
            System.out.println(line);
        }
        boolean errorFound = false;
        String errorString = "";
        while ((line = bufferedReaderError.readLine()) != null) {
            errorString += line + '\n';
            errorFound = true;
        }

        if (errorFound) {
            System.err.println(errorString);
            throw new InterruptedIOException("Error while executing CARSKit");
        }

        //Wait to get exit value
        int exitValue = process.waitFor();
        System.out.println("\n\nExit Value is " + exitValue);
    }


    public static void main(String[] args) throws IOException, InterruptedException {

        String ratingsFolder =
                "/Users/fpena/UCC/Thesis/datasets/context/stuff/cache_context/" +
                        "carskit/yelp_hotel_carskit_ratings_ensemble_numtopics-30" +
                        "_iterations-100_passes-10_targetreview-specific_" +
                        "normalized_ck-no_context_lang-en_bow-NN_" +
                        "document_level-review_targettype-context_" +
                        "min_item_reviews-10/";
        String algorithm = "camf_ci";
        String confFile = ratingsFolder + algorithm + ".conf";

        CarskitCaller.run(confFile);
    }
}
