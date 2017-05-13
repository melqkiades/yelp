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

import net.recommenders.rival.core.DataModel;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Temporal data model used throughout the toolkit. It is able to store users,
 * items, preferences, and timestamps.
 *
 * @author franpena
 *
 * @param <U> generic type for users
 * @param <I> generic type for items
 */
public class ContextDataModel<U, I> extends DataModel<U, I> implements ContextDataModelIF<U, I> {

    /**
     * The map with the timestamps between users and items.
     */
    protected Map<U, Map<I, Map<String, Double>>> userItemContexts;

    /**
     * Default constructor.
     */
    public ContextDataModel() {
        super();
        userItemContexts = new HashMap<>();
    }

    /**
     * Constructor with parameters.
     *
     * @param ignoreDupPreferences The flag to indicate whether preferences
     * should be ignored.
     */
    public ContextDataModel(final boolean ignoreDupPreferences) {
        super(ignoreDupPreferences);
        userItemContexts = new HashMap<>();
    }

    /**
     * Constructor with parameters.
     *
     * @param ignoreDupPreferences The flag to indicate whether preferences
     * should be ignored.
     * @param userItemPreference The preference map between users and items.
     * @param itemSet The items.
     * @param userItemContext The map with the timestamps between users and
     * items
     */
    public ContextDataModel(final boolean ignoreDupPreferences, final Map<U, Map<I, Double>> userItemPreference, final Set<I> itemSet,
                            final Map<U, Map<I, Map<String, Double>>> userItemContext) {
        super(ignoreDupPreferences, userItemPreference, itemSet);
        this.userItemContexts = userItemContext;
    }

    /**
     * Method that returns the map with the timestamps between users and items.
     *
     * @return the map with the timestamps between users and items.
     */
    @Override
    public Map<U, Map<I, Map<String, Double>>> getUserItemContext() {
        return userItemContexts;
    }

    /**
     * Method that adds a timestamp to the model between a user and an item.
     *
     * @param u the user.
     * @param i the item.
     * @param c the context.
     */
    @Override
    public void addContext(final U u, final I i, final Map<String, Double> c) {
        Map<I, Map<String, Double>> userContexts = userItemContexts.get(u);
        if (userContexts == null) {
            userContexts = new HashMap<>();
            userItemContexts.put(u, userContexts);
        }
        userContexts.put(i, c);
    }

    /**
     * Method that clears all the maps contained in the model.
     */
    public void clear() {
        super.clear();
        userItemContexts.clear();
    }
}
