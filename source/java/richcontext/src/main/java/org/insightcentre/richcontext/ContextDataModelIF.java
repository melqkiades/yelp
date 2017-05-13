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

import java.util.Map;
import java.util.Set;

/**
 * Interface for a contextual data model. It is able to store users, items,
 * preferences, and contextual information.
 *
 * @author franpena
 *
 * @param <U> generic type for users
 * @param <I> generic type for items
 */
public interface ContextDataModelIF<U, I> extends DataModelIF<U, I> {

    /**
     * Method that returns the map with the timestamps between users and items.
     *
     * @return the map with the timestamps between users and items.
     */
    public Map<U, Map<I, Map<String, Double>>> getUserItemContext();

    /**
     * Method that adds a preference to the model between a user and an item
     * under the context c.
     *
     * @param u the user.
     * @param i the item.
     * @param c the context.
     */
    public void addContext(final U u, final I i, final Map<String, Double> c);
}
