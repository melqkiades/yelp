package org.insightcentre.richcontext;

import java.util.function.Supplier;

/**
 * Created by fpena on 19/09/2018.
 */
public enum RecommenderLibrary {
    LIBFM("libfm", "libfm", LibfmExporter::new),
    CARSKIT("carskit", "csv", CarskitExporter::new);

    private String name;
    private String extension;
    private Supplier<ReviewsExporter> instantiator;

    RecommenderLibrary(String name, String extension, Supplier<ReviewsExporter> instantiator) {
        this.name = name;
        this.extension = extension;
        this.instantiator = instantiator;
    }

    public String getName() {
        return name;
    }

    public String getExtension() {
        return extension;
    }

    public ReviewsExporter getReviewsExporter() {
        return instantiator.get();
    }
}
