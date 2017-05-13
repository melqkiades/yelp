package org.insightcentre.richcontext;

import com.google.gson.annotations.SerializedName;

import java.util.Map;

/**
 * Created by fpena on 23/03/2017.
 */
public class Review {

    @SerializedName("review_id")
    private String id;

    @SerializedName("item_integer_id")
    private final long itemId;

    @SerializedName("user_integer_id")
    private final long userId;

    @SerializedName("stars")
    private double rating;

    private double predictedRating;

    @SerializedName("predicted_class")
    private String predictedClass;

    @SerializedName("topic_model_target")
    private String topicModelTarget;

    @SerializedName("sentence_index")
    private int sentenceIndex;

    @SerializedName("language")
    private String language;

    @SerializedName("has_context")
    private boolean hasContext;

    @SerializedName("user_item_integer_key")
    private String user_item_key;

//    @SerializedName("bow")
//    private List<String> bagOfWords;

    @SerializedName("context")
    private Map<String, Double> context;


    public Review(long userId, long itemId) {
        this.userId = userId;
        this.itemId = itemId;
    }

    public Review(long userId, long itemId, double rating) {
        this.userId = userId;
        this.itemId = itemId;
        this.rating = rating;
    }

    public String getId() {
        return id;
    }

    public long getItemId() {
        return itemId;
    }

    public long getUserId() {
        return userId;
    }

    public double getRating() {
        return rating;
    }

    public String getUser_item_key() {
        return userId + "|" + itemId;
    }

    public Map<String, Double> getContext() {
        return context;
    }

    public void setContext(Map<String, Double> context) {
        this.context = context;
    }

    public double getPredictedRating() {
        return predictedRating;
    }

    public void setPredictedRating(double predictedRating) {
        this.predictedRating = predictedRating;
    }

    @Override
    public String toString() {
        return "Review{" +
                "itemId=" + itemId +
                ", userId=" + userId +
                ", rating=" + rating +
                ", predictedRating=" + predictedRating +
                ", context=" + context +
                '}';
    }
}
