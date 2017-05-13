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

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Data model used throughout the toolkit. Able to store users, items,
 * preferences, timestamps.
 *
 * @author <a href="http://github.com/abellogin">Alejandro</a>, <a
 * href="http://github.com/alansaid">Alan</a>
 *
 * @param <U> generic type for users
 * @param <I> generic type for items
 */
public class NewContextDataModel<U, I> {

    /**
     * Preference map between users and items.
     */
    protected Map<U, Map<Map<String, Double>, Map<I, Double>>> userContextItemPreferences;
    /**
     * Set containing all the items.
     */
    protected Set<I> items;
    protected Set<Map<String, Double>> contexts;

    /**
     * Default constructor.
     */
    public NewContextDataModel() {

        this.items =  new HashSet<>();
        this.contexts = new HashSet<>();
        this.userContextItemPreferences = new HashMap<>();
    }

    /**
     * Method that returns the preference map between users and items.
     *
     * @return the preference map between users and items.
     */
    public Map<U, Map<Map<String, Double>, Map<I, Double>>> getUserContextItemPreferences() {
        return userContextItemPreferences;
    }

    /**
     * Method that adds a preference to the model between a user and an item.
     *
     * @param u the user.
     * @param i the item.
     * @param d the preference.
     */
    public void addPreference(final U u, final I i, final Map<String, Double> c, final Double d) {
        // update direct map
        Map<Map<String, Double>, Map<I, Double>> userPreferences = userContextItemPreferences.get(u);
        if (userPreferences == null) {
            userPreferences = new HashMap<>();
            userContextItemPreferences.put(u, userPreferences);
        }

        Map<I, Double> userContextPreferences = userPreferences.get(c);
        if (userContextPreferences == null) {
            userContextPreferences = new HashMap<>();
            userPreferences.put(c, userContextPreferences);
        }

        userContextPreferences.put(i, d);
        // update items
        items.add(i);
        contexts.add(c);
    }

    /**
     * Method that returns the items in the model.
     *
     * @return the items in the model.
     */
    public Set<I> getItems() {
        return items;
    }

    /**
     * Method that returns the users in the model.
     *
     * @return the users in the model.
     */
    public Set<U> getUsers() {
        return getUserContextItemPreferences().keySet();
    }

    public Set<Map<String, Double>> getContexts() {
        return contexts;
    }

    /**
     * Method that returns the number of items in the model.
     *
     * @return the number of items in the model.
     */
    public int getNumItems() {
        return items.size();
    }


    public int getNumContexts() {
        return contexts.size();
    }

    /**
     * Method that returns the number of users in the model.
     *
     * @return the number of users in the model.
     */
    public int getNumUsers() {
        return getUsers().size();
    }

    /**
     * Method that clears all the maps contained in the model.
     */
    public void clear() {
        userContextItemPreferences.clear();
        items.clear();
        contexts.clear();
    }


    public void saveDataModel(final String outfile, final boolean overwrite, final String delimiter)
            throws FileNotFoundException, UnsupportedEncodingException {
        if (new File(outfile).exists() && !overwrite) {
            System.out.println("Ignoring " + outfile);
        } else {
            PrintStream out = new PrintStream(outfile, "UTF-8");
            out.println("user_id" + delimiter + "item_id" + delimiter + "rating");
            for (U user : this.getUsers()) {

                Map<Map<String, Double>, Map<I, Double>> userContextPreferences =
                        this.getUserContextItemPreferences().get(user);
                Map<String, Double> context = userContextPreferences.keySet().iterator().next();
                Map<I, Double> itemPreferences = userContextPreferences.get(context);

                for (Map.Entry<I, Double> entry : itemPreferences.entrySet()) {
                    I item = entry.getKey();
                    Double pref = itemPreferences.get(item);
                    out.println(user + delimiter + item + delimiter + pref);
                }

//                for (Map.Entry<I, Double> e : userPrefModel.entrySet()) {
//                    I item = e.getKey();
//                    Double pref = userPrefModel.get(item);
//                    out.println(user + delimiter + item + delimiter + pref);
//                }
            }
            out.close();
        }
    }
}
