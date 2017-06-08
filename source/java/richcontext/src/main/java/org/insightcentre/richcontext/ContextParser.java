/*
 * Copyright 2015 recommenders.net.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.insightcentre.richcontext;

import net.recommenders.rival.core.DataModelIF;
import net.recommenders.rival.core.Parser;
import net.recommenders.rival.core.TemporalDataModel;
import net.recommenders.rival.core.TemporalDataModelIF;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;

/**
 * Data parser for tab-separated data files.
 *
 * @author <a href="http://github.com/abellogin">Alejandro</a>
 */
public class ContextParser {

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
     * Parses a data file with a specific separator between fields.
     *
     * @param f The file to be parsed.
     * @param token The separator to be used.
     * @return A dataset created from the file.
     * @throws IOException if the file cannot be read.
     */
    public NewContextDataModel<Long, Long> parseData(final File f, final String token) throws IOException {
        NewContextDataModel<Long, Long> dataset = new NewContextDataModel<>();

        BufferedReader br = ContextParser.getBufferedReader(f);
        String line = br.readLine();
        if ((line != null) && (!line.matches(".*[a-zA-Z].*"))) {
            parseLine(line, dataset, token);
        }
        while ((line = br.readLine()) != null) {
            parseLine(line, dataset, token);
        }
        br.close();

        return dataset;
    }


    public NewContextDataModel<Long, Long> parseData2(final File f, final String token) throws IOException {

        NewContextDataModel<Long, Long> dataset = new NewContextDataModel<>();

        BufferedReader br = ContextParser.getBufferedReader(f);

        String testLine;

        // Parse the header
        testLine = br.readLine();
        String[] headers = testLine.split("\t");

        while ((testLine = br.readLine()) != null) {
            String[] lineTokens = testLine.split("\t");
            long user_id = Long.parseLong(lineTokens[USER_TOK]);
            long item_id = Long.parseLong(lineTokens[ITEM_TOK]);
            double predictedRating = Double.parseDouble(lineTokens[RATING_TOK]);
//            Map<String, Double> contextMap = new HashMap<>();

//            for (int i = 3; i < lineTokens.length; i++) {
//                contextMap.put(headers[i], Double.parseDouble(lineTokens[i]));
//            }

            dataset.addPreference(user_id, item_id, null, predictedRating);
        }

        return dataset;
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
    private void parseLine(final String line, final NewContextDataModel<Long, Long> dataset, final String token) {
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
//        dataset.addPreference(userId, itemId, preference);
    }
}
