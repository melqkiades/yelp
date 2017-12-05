package org.insightcentre.richcontext;

import com.opencsv.CSVWriter;
import net.recommenders.rival.core.DataModelIF;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by fpena on 20/07/2017.
 */
public class Utils {


    /**
     * Returns the path of the folder where the data for RiVal is going to be
     * stored. If necessary, creates the required directories
     *
     * @param jsonRatingsFile the path of the file that contains the original
     * dataset to be evaluated
     * @return the path of the folder where the data for RiVal is going to be
     * stored
     */
    public static String getRatingsFolderPath(String jsonRatingsFile) throws IOException {

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

        String ratingsFolderPath = rivalFolderPath + jsonFileName + "/";

        File ratingsDir = new File(ratingsFolderPath);
        if (!ratingsDir.exists()) {
            if (!ratingsDir.mkdir()) {
                throw new IOException("Directory " + ratingsDir + " could not be created");
            }
        }

        System.out.println("File name: " + jsonFileName);
        System.out.println("Parent folder: " + jsonFileParentFolder);
        System.out.println("Ratings folder: " + ratingsFolderPath);

        return ratingsFolderPath;
    }


    /**
     * Writes the given {@code Map} of results into the {@code outputFile}
     * using CSV format. If the file already exists, the results will be added
     * into a new line at the end of the file
     *
     * @param resultsList the evaluation results after running the recommender
     * @param outputFile the name of the file that is going to contain the
     * results
     * @param headers the headers of the CSV file
     * @throws IOException
     */
    public static void writeResultsToFile(
            List<Map<String, String>> resultsList, String outputFile,
            String[] headers)
            throws IOException {

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
}
