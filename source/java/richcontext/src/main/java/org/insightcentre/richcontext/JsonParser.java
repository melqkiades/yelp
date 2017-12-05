package org.insightcentre.richcontext;

import com.google.gson.Gson;
import net.recommenders.rival.core.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import net.recommenders.rival.core.TemporalDataModelIF;


/**
 * Created by fpena on 23/03/2017.
 */
public class JsonParser implements Parser<Long, Long> {


    private List<Review> reviews;



    public static void main(String[] args) throws IOException {

        String folder = "/Users/fpena/UCC/Thesis/datasets/context/stuff/cache_context/";
//        String filePath = folder + "yelp_restaurant_recsys_contextual_records_ensemble_numtopics-10_iterations-100_passes-10_targetreview-specificnormalized_lang-en_bow-NN_document_level-review_targettype-context_min_item_reviews-10.json";
        String filePath = folder + "yelp_hotel_recsys_formatted_context_records_ensemble_numtopics-10_iterations-100_passes-10_targetreview-specific_normalized_contextformat-no_context_lang-en_bow-NN_document_level-review_targettype-context_min_item_reviews-10.json";

        File jsonFile = new File(filePath);
        System.out.println(readReviews(jsonFile).get(10));
        System.out.println(readReviews(jsonFile).get(10).getContext());
    }



    public static List<Review> readReviews(File file) throws IOException {

        Gson gson = new Gson();
        List<Review> reviews = new ArrayList<>();

        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        while ((line = br.readLine()) != null) {
            Review review = gson.fromJson(line, Review.class);
            reviews.add(review);
        }

        System.out.println("Number of reviews: " + reviews.size());

        return reviews;
    }

    @Override
    public TemporalDataModelIF<Long, Long> parseTemporalData(File f) throws IOException {
        throw new UnsupportedOperationException("Method not yet implemented");
    }

    @Override
    public DataModelIF<Long, Long> parseData(File f) throws IOException {

        DataModelIF<Long, Long> dataset = new TemporalDataModel<>();
        this.reviews = readReviews(f);

        for (Review review : this.reviews) {
            dataset.addPreference(
                    review.getUserId(), review.getItemId(), review.getRating()
            );
        }

        return dataset;
    }

    public List<Review> getReviews() {
        return reviews;
    }
}
