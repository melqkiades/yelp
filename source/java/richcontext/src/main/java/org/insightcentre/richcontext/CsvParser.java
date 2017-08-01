package org.insightcentre.richcontext;

/**
 * Created by fpena on 11/07/2017.
 */

import me.tongfei.progressbar.ProgressBar;
import me.tongfei.progressbar.ProgressBarStyle;
import net.recommenders.rival.core.DataModelIF;
import net.recommenders.rival.core.TemporalDataModel;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;
import java.util.zip.GZIPInputStream;

/**
 * Data parser for tab-separated data files.
 *
 * @author <a href="http://github.com/abellogin">Alejandro</a>
 */
public class CsvParser {

    /**
     * The column index for the user id in the file.
     */
    public static final int USER_TOK = 0;
    /**
     * The column index for the item id in the file.
     */
    public static final int ITEM_TOK = 1;
    /**
     * The column index for the rating in the file.
     */
    public static final int RATING_TOK = 2;
    /**
     * The column index for the time in the file.
     */
    public static final int TIME_TOK = 3;

    /**
     * {@inheritDoc}
     */
    public DataModelIF<Long, Long> parseData(final File f, Set<Long> users)
            throws IOException {
        return parseData(f, "\t", true, users);
    }

    /**
     * Parses a data file with a specific separator between fields.
     *
     * @param f The file to be parsed.
     * @param token The separator to be used.
     * @return A dataset created from the file.
     * @throws IOException if the file cannot be read.
     */
    public DataModelIF<Long, Long> parseData(
            final File f, final String token,
            final boolean ignoreDupPreferences, final Set<Long> users)
            throws IOException {
        DataModelIF<Long, Long> dataset = new TemporalDataModel<>(ignoreDupPreferences);

        BufferedReader br = CsvParser.getBufferedReader(f);
        String line = br.readLine();
        if ((line != null) && (!line.matches(".*[a-zA-Z].*"))) {
            parseLine(line, dataset, token, users);
        }
        while ((line = br.readLine()) != null) {
            parseLine(line, dataset, token, users);
        }
        br.close();

        return dataset;
    }

    public DataModelIF<Long, Long> parseData(
            final File f, final String token, final Set<Long> users)
            throws IOException {

        return parseData(f, token, true, users);
    }


    /**
     * Parses a data file with a specific separator between fields.
     *
     * @param f The file to be parsed.
     * @param token The separator to be used.
     * @return A dataset created from the file.
     * @throws IOException if the file cannot be read.
     */
    public DataModelIF<Long, Long> parseData(
            final File f, final String token, final boolean ignoreDupPreferences) throws IOException {
        DataModelIF<Long, Long> dataset = new TemporalDataModel<>(ignoreDupPreferences);

        BufferedReader br = CsvParser.getBufferedReader(f);
        long numLines = Files.lines(Paths.get(f.getAbsolutePath())).count();
        ProgressBar progressBar = new ProgressBar(
                "Parsing CSV data", numLines, ProgressBarStyle.ASCII);
        progressBar.start();
        String line = br.readLine();
        if ((line != null) && (!line.matches(".*[a-zA-Z].*"))) {
            parseLine(line, dataset, token);
        }
        while ((line = br.readLine()) != null) {
            parseLine(line, dataset, token);
            progressBar.step();
        }
        br.close();
        progressBar.stop();
        System.out.println("\n");

        return dataset;
    }


    /**
     * Parses a data file with a specific separator between fields. It ignores
     * duplicate preferences, i.e., it only incorporates the first occurrence
     * of a user-item pair, ignoring the subsequent ratings for the same
     * user-item pair.
     *
     * @param f The file to be parsed.
     * @param token The separator to be used.
     * @return A dataset created from the file.
     * @throws IOException if the file cannot be read.
     */
    public DataModelIF<Long, Long> parseData(
            final File f, final String token) throws IOException {

        return parseData(f, token,  true);
    }

    /**
     * {@inheritDoc}
     */
    public DataModelIF<Long, Long> parseData(final File f) throws IOException {
        return parseData(f, "\t", true);
    }


    /**
     * Parses line from data file.
     *
     * @param line The line to be parsed.
     * @param dataset The dataset to add data from line to.
     * @param token the token to split on.
     */
    private void parseLine(
            final String line, final DataModelIF<Long, Long> dataset,
            final String token) {
        if (line == null) {
            return;
        }
        String[] toks = line.split(token);
        // user
        long userId = Long.parseLong(toks[USER_TOK]);
        // item
        long itemId = Long.parseLong(toks[ITEM_TOK]);
        // preference
        double preference = Double.parseDouble(toks[RATING_TOK]);
        //////
        // update information
        //////
        dataset.addPreference(userId, itemId, preference);
    }


    /**
     * Obtains an instance of BufferedReader depending on the file extension: if
     * it ends with gz, zip, or tgz then a compressed reader is used instead of
     * the standard one.
     *
     * @param f The file to be opened.
     * @return An instance of BufferedReader or null if there is a problem
     * @throws IOException when the file cannot be read.
     * @see BufferedReader
     */
    public static BufferedReader getBufferedReader(final File f) throws IOException {
        BufferedReader br = null;
        if ((f == null) || (!f.isFile())) {
            return br;
        }
        if (f.getName().endsWith(".gz") || f.getName().endsWith(".zip") || f.getName().endsWith(".tgz")) {
            br = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(f)), "UTF-8"));
        } else {
            br = new BufferedReader(new InputStreamReader(new FileInputStream(f), "UTF-8"));
        }
        return br;
    }

    /**
     * Parses line from data file.
     *
     * @param line The line to be parsed.
     * @param dataset The dataset to add data from line to.
     * @param token the token to split on.
     */
    private void parseLine(
            final String line, final DataModelIF<Long, Long> dataset,
            final String token, final Set<Long> users) {
        if (line == null) {
            return;
        }
        String[] toks = line.split(token);
        // user
        long userId = Long.parseLong(toks[USER_TOK]);

        if (!users.contains(userId)) {
            return;
        }

        // item
        long itemId = Long.parseLong(toks[ITEM_TOK]);
        // preference
        double preference = Double.parseDouble(toks[RATING_TOK]);
        // timestamp
        long timestamp = -1;
        //////
        // update information
        //////
        dataset.addPreference(userId, itemId, preference);
    }


    public static void main(String[] args) throws IOException {

        String folder = "/Users/fpena/UCC/Thesis/datasets/context/stuff/" +
                "cache_context/rival/yelp_hotel_recsys_formatted_context_" +
                "records_ensemble_numtopics-10_iterations-100_passes-10_" +
                "targetreview-specific_normalized_contextformat-no_context_" +
                "lang-en_bow-NN_document_level-review_targettype-context_min_" +
                "item_reviews-10/fold_0/";

        String fileName = folder + "strategymodel_libfm_REL_PLUS_N.csv";

        Set<Long> users = new HashSet<>();
        users.add(262L);
        users.add(615L);

        CsvParser parser = new CsvParser();
        DataModelIF<Long, Long> dataModel = parser.parseData(new File(fileName), users);
        System.out.println("Num users = " + dataModel.getNumUsers());
        System.out.println("Num items = " + dataModel.getNumItems());
        System.out.println("Users: " + dataModel.getUsers());
        System.out.println("Items: " + dataModel.getItems());
    }
}
