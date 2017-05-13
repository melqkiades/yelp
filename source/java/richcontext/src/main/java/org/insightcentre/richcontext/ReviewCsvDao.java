package org.insightcentre.richcontext;

import com.opencsv.CSVReader;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by fpena on 27/03/2017.
 */
public class ReviewCsvDao {


    private static final int USER_ID_INDEX = 0;
    private static final int ITEM_ID_INDEX = 1;
    private static final int RATING_INDEX = 2;

    private static final String separator = "\t";


    public static List<Review> readCsvFile(String filePath)
            throws IOException {

        CSVReader reader = new CSVReader(new FileReader(filePath), '\t');
        List<Review> reviews = new ArrayList<>();

        String [] nextLine;
        while ((nextLine = reader.readNext()) != null) {
            long user_id = Long.parseLong(nextLine[USER_ID_INDEX]);
            long item_id = Long.parseLong(nextLine[ITEM_ID_INDEX]);
            double rating = Double.parseDouble(nextLine[RATING_INDEX]);
            Review review = new Review(user_id, item_id, rating);
            reviews.add(review);
        }

        return reviews;
    }

    public static void main(String[] args) {

        String file_path = "/Users/fpena/tmp/rival/data/rich-context/model/train_0.csv";
        try {
            List<Review> reviews = readCsvFile(file_path);
            for (Review review : reviews) {
                System.out.println(review);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
