import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import itertools
from matplotlib import rcParams
import seaborn as sns
from scipy.stats import ttest_ind
from group_visualisations import *

# figure size in inches
rcParams['figure.figsize'] = 8, 5


def get_positive_quotes(df):
    """
    Extract all the positive quotes from dataframe
    Parameters:
      - df: Dataframe containing all quotes
    Returns: 
      - Dataframe containing only the positive quotes
    """
    return df[df['sentiment'] == 1]


def get_negative_quotes(df):
    """
    Extract all the negative quotes from dataframe
    Parameters:
      - df: Dataframe containing all quotes
    Returns: 
      - Dataframe containing only the negative quotes
    """
    return df[df['sentiment'] == -1]


def get_neutral_quotes(df):
    """
    Extract all the neutral quotes from dataframe
    Parameters:
      - df: Dataframe containing all quotes
    Returns: 
      - Dataframe containing only the neutral quotes
    """
    return df[df['sentiment'] == 0]


def statistical_difference(df1, df2, alpha=0.05):
    """
    Test if the means of the value in the two dataframe comes from differnet population 
    Parameters:
      - df1 : First sample population
      - df2 : second sample population
      - alpha: Significance level of the test
    Returns: 
      - p_value: the p_value of the test
      - alpha: the significance level of the test
      - significant: if the test is significant at level alpha or not
    """
    stat, p_value = ttest_ind(df1, df2)
    return p_value, alpha, alpha > p_value


def print_statistical_test_result(p_value, alpha, significant):
    """
    Print the result of the test
    Parameters: 
      - p_value: the p_value of the test
      - alpha: the significance level of the test
      - significant: if the test is significant at level alpha or not
    """
    if significant:
        print(f"The difference between the two means is significant at level {alpha:.2f} (p-value = {p_value:.3e})")
    else:
        print(f"The test is not significant at level {alpha:.2f}")


def print_all_shapes(dfs, titles):
    """
    Print the shapes of all dataframe in the list
    Parameters:
      - dfs: List of dataframes
      - titles: The name of all dataframes 
    """
    for df, title in zip(dfs, titles):
        print("Shape of " + title + f" dataframe: {df.shape}")


def count_nb_none_rows(df):
    """
    Print the number of rows in the dataframe and the number of rows with none value for each column
    Parameters: 
      - df : the dataframe from which we want to comptue the counts
    """
    print(f"\nTotal number of rows in dataframe: {df.shape[0]}")
    attributes = ['speaker_gender', "speaker_nationality", "speaker_ethnic_group", "speaker_occupation",
                  "speaker_party", "speaker_academic_degree", "speaker_candidacy", "speaker_religion"]
    for attr in attributes:
        count = df.apply(lambda x: x[attr] is None, axis=1).sum()
        print(f"Number of None in {attr} : {count}")


def load_dataframe_and_extract_sentiment(file, neg_threshold=-1 / 3, pos_threshold=1 / 3):
    """
    Load the dataframes with sentiment, attribute a sentiment to each quote and remove people over 125 years
    Parameters:
        file: The file where the dataframe is
        neg_threshold: the threshold under which the quote is classified as negative
        pos_threshold: the threshold above which the quote is classified as positive

    Returns:
        The loaded dataframe
    """
    # Load the compressed dataframe
    df = pd.read_json(file, lines=True, compression='bz2')

    # Classify the sentiment according to the given thresholds
    df['sentiment'] = df.apply(
        lambda row: -1 if row['compound'] <= neg_threshold else 1 if row['compound'] >= pos_threshold else 0, axis=1)

    # Remove person without any age or age > 125
    df = remove_too_old_people_and_na(df)
    return df


def remove_too_old_people_and_na(df, age_threshold=125):
    """
    Remove people whose age is above 125 or undefined
    Args:
        df: Dataframe containing all the quotes
        age_threshold: The age above which we will remove the quotes

    Returns:
        The cleaned dataframe
    """
    df = df[~df['speaker_age'].isna()]
    return df[~(df['speaker_age'] > age_threshold)]


"""
Clustering related functions
"""


def k_medioid(k, distance_matrix, seed=0, verbose=True, min=1):
    """
    Perform a custom version of the K-Medioid. 
    
    Parameters:
      - k : the maximal number of clusters the algorithm can make
      - distance_matrix: the pairwise distance between any two person
      - seed : seed to set for reproducibility
      - verbose: A boolean indicating whether we display epoch information or not
      - min: the minimal number of people in each cluster. 
    Returns:
      center: the cluster centers.
      assignment: the assignment of cluster fo reach data point.
      k : the number of cluster created. 
    """
    # Set the seed for reproducibility
    np.random.seed(seed)
    # Randomly initialise the cluster center
    center = np.random.choice(len(distance_matrix), k, replace=False)
    cpt = 0
    assignment = None
    redo = True

    while redo:
        old_center = np.ones(k) * (-1)
        while np.any(center != old_center):
            # Loop while the center cluster are still moving
            cpt += 1
            if verbose:
                print(f"\rEpoch {cpt}", end='')

            old_center = center.copy()
            # Assign the points to the closest center
            assignment = distance_matrix[list(center)].argmin(axis=0)

            for i in range(k):
                # Define the new centroid by taking the minimal sum of distance in the cluster
                mask = assignment == i
                if (mask.any()):
                    center[i] = np.where(mask, distance_matrix[mask].sum(axis=0), float("inf")).argmin()
                    assert (mask[center[i]])

        redo = False
        remove = (-1, min)
        for i in range(k):
            # Check if the number of points assigned to the cluster is smaller than the the current minimal.
            nb = len(assignment[assignment == i])
            if nb <= remove[1]:
                remove = (i, nb)

        # If we have at least one cluster with a smaller number of points than theminimal required, loop again.
        if remove[0] >= 0:
            center = np.delete(center, [remove[0]])
            redo = True

        k = len(center)

    if verbose:
        # Print the new value of k
        print(f"\nk : {k}")
    return center, assignment, k


def get_unique_person_attributes(df, attributes):
    """
    Get a dataframe with all the unique people in the quotes selecting only the given attributes
    Parameters: 
    -----------
      - df : The dataframe containing all the quotes
      - attributes : all the attributse we need 
    Returns:
    -----------
      Return the unique people dataframe
    """
    df['Author'] = df.apply(lambda row: row['qid'], axis=1)
    compound = df.groupby('Author').apply(lambda row: row['compound'].mean())
    filter = df['Author'].duplicated(keep='first')
    df = df[~filter]
    df = df[attributes]
    df['mean_compound'] = df['Author'].apply(lambda row: compound[row])
    df = df.reset_index(drop=True)
    return df


def jaccard_distance(set1, set2):
    """
    Compute the Jaccard distance between the two given sets
    Parameters:
      - set1, set2: The two sets from which we want to compute the distance
    Returns:
      - the Jaccard distance
    """
    return 1 - len(set1.intersection(set2)) / len(set1.union(set2))


def build_distance_matrix(df, attributes, use_mean_compound=False):
    """
    Build the pairwise distance between any two people in the dataframe. 
    Parameters: 
      - df: The dataframe containing attributes of people
      - attributes: The list of attributes we need to consider to compute the distance 
      - use_mean_compound: A boolean indicating whether we should use the mean_compound score of quotes or not
    Returns: 
      - The matrix with the pairwise distance
    """
    distance = np.zeros((df.shape[0], df.shape[0]), dtype=np.single)
    max_age = df.speaker_age.max()
    min_age = df.speaker_age.min()

    def distance_between_persons(df1, df2):
        """
        Compute the distance between two persons
        Parameters:
            df1: The attributes of the first person
            df2: The attributes of the second person
        """
        if df1.name < df2.name:
            dist = 0
            if pd.isna(df1['speaker_age']) or pd.isna(df2['speaker_age']):
                dist += 1
            else:
                # Rescale the age difference to the [0,1] range so that the importance of the age attribute is the
                # same as the other attributes
                dist += np.abs(df1['speaker_age'] - df2['speaker_age']) / (max_age - min_age)

            if use_mean_compound:
                # Give a bit more importance to the difference of mean compound score.
                dist += np.abs(df1['mean_compound'] - df2['mean_compound'])

            for attr in attributes:
                if df1[attr] is None:
                    dist += 1 if df2[attr] is not None else 0
                elif df2[attr] is None:
                    # set1 is not None
                    dist += 1
                else:
                    # Both sets are not None
                    dist += jaccard_distance(set(df1[attr]), set(df2[attr]))

            # Since the matrix is a distance matrix, it should be symmetric and therefore we avoid double computation
            distance[df1.name, df2.name] = dist
            distance[df2.name, df1.name] = dist

    # Apply the function to the dataframe
    df.apply(lambda x: df.apply(lambda y: distance_between_persons(x, y), axis=1), axis=1)

    return distance


def get_unique_people_dist_save(df, unique_people_path, distance_matrix_path, advanced_filtering=True,
                                use_mean_compound=False):
    """
    Get a dataframe with all the unique people in the quotes selecting only the given attributes
    Build/Load the pairwise distance between any two people in the dataframe. 
    Parameters: 
      - df: The dataframe containing attributes of people
      - unique_people_path : path of unique_people dataframe
      - distance_matrix_path : path of distance_matrix_path dataframe
      - advanced_filtering : if we filter the people with no age, nationality, gender or occupation
      - use_mean_compound: A boolean indicating whether we should use the mean_compound score of quotes or not
    Returns: 
      - df_unique_people : the unique people dataframe
      - distance_matrix : The matrix with the pairwise distance
    """
    if isfile(unique_people_path) and isfile(distance_matrix_path):
        distance_matrix = np.load(distance_matrix_path)
        df_unique_people = pd.io.json.read_json(unique_people_path, encoding='utf-8')
    else:
        df_unique_people = get_unique_person_attributes(df, ['Author', 'speaker', 'speaker_age', 'speaker_gender',
                                                             'speaker_nationality', 'speaker_occupation',
                                                             'speaker_religion', 'speaker_continent'])

        if advanced_filtering:
            # it can happen that we have too many people and for computational and result purposes, we keep only the
            # author of quotes that have their attributes set
            mask = df_unique_people.apply(
                lambda x: (x["speaker_nationality"] is None) or (pd.isna(x['speaker_age'])) or (
                        x['speaker_gender'] is None) or (x['speaker_occupation'] is None), axis=1)
            df_unique_people = df_unique_people[~mask]

        distance_matrix = build_distance_matrix(df_unique_people,
                                                ['speaker_gender', 'speaker_nationality', 'speaker_occupation',
                                                 'speaker_religion', 'speaker_continent'], use_mean_compound)

        df_unique_people.to_json(unique_people_path, encoding='utf-8')
        np.save(distance_matrix_path, distance_matrix)

    return df_unique_people, distance_matrix


def remove_quotes_without_author_in_distance_matrix(df_quotes, df_unique_people):
    """
    Remove quotes whose author is not in the computed distance matrix
    Args:
        df_quotes: dataframe with all the quotes
        df_unique_people: dataframe containing all the people used when computing the distance matrix

    Returns:
        The cleaned quotes dataframe
    """
    # Remove quotes whose author is not in the computed distance matrix
    mask = df_quotes.apply(lambda x: len(df_unique_people.index[df_unique_people['Author'] == x['qid']]) == 0, axis=1)
    filtered_quotes = df_quotes[~mask]
    return filtered_quotes


"""
Cluster investigating functions
"""


def assign_quotes_to_cluster(assignments, df_quotes, df_unique_people):
    """
    Assign a cluster ID to each quotes
    Parameters:
        assignments: an array where each component is an assignment to a cluster
        df_quotes: the dataframe with all the quotes
        df_unique_people: the dataframe with all the unique people in the quotes

    Returns:
        The dataframe with one more column corresponding to the cluster assigned to each quotes
    """
    df_quotes['assigned_cluster'] = df_quotes.apply(
        lambda x: assignments[df_unique_people.index[df_unique_people['Author'] == x['qid']][0]], axis=1)
    return df_quotes


def assign_people_to_cluster(assignments, df_people):
    """
    Assign a cluster ID to each person
    Parameters:
        assignments: an array where each component is an assignment to a cluster
        df_people: the dataframe with all the unique people in the quotes

    Returns:
        The dataframe with one more column corresponding to the cluster assigned to each person
    """
    df_people['assigned_cluster'] = df_people.apply(lambda x: assignments[x.name], axis=1)
    return df_people


def reorder_clusters(cat_by_clusters, df, df_unique_people, k):
    """
    Reorder the clusters id in the order corresponding to the mean
    Args:
        cat_by_clusters: the different mean score by clusters
        df: dataframes containing all the quotes
        df_unique_people: dataframe with all the people talking about the investigated person

    Returns:
        The dataframes with updated cluster order
            cat_by_clusters, df, df_unique_people
    """
    # Reorder the cluster by proportion of positive sentiment in the quotes (to make plot more beautiful)
    order = list(cat_by_clusters[cat_by_clusters['sentiment'] == 1].sort_values(['prop'], ascending=False)[
                     'assigned_cluster'].values)
    sorter = np.zeros(k, dtype=int)
    for i in range(k):
        if i not in order:
            order.append(i)
        sorter[order[i]] = i

    # Relabel the clusters
    cat_by_clusters['assigned_cluster'] = cat_by_clusters['assigned_cluster'].apply(lambda x: sorter[int(x)])
    df['assigned_cluster'] = df['assigned_cluster'].apply(lambda x: sorter[int(x)])
    df_unique_people['assigned_cluster'] = df_unique_people['assigned_cluster'].apply(lambda x: sorter[int(x)])
    return cat_by_clusters, df, df_unique_people


def get_pos_neg_ratio_in_groups(df, df_unique_people, k):
    """
    Assign the cluster number to each author and plot various statistics
    Parameters:
      - df: The dataframe containing all the quotes
      - df_unique_people: The dataframe containing all the unique people
      - assignments: The assignment of people to cluster
      - k: The number of cluster
    Returns:
      - cat_by_clusters: the different mean score by clusters
      - df: the dataframe of quotes with the assigned cluster
      - df_unique_people: the dataframe of people with the assigned cluster
    """

    # Compute the sentiments from the compound score
    sentiments = df.apply(lambda row: -1 if row['compound'] <= -1 / 3 else 1 if row['compound'] >= 1 / 3 else 0, axis=1)

    # Count the number of quotes falling in each (clusterId, sentiment) pairs
    cat_by_clusters = sentiments.groupby(df.assigned_cluster).value_counts()
    cat_by_clusters = cat_by_clusters.reset_index().rename(columns={0: 'count', 'level_1': 'sentiment'})

    # Sum the value in each cluster
    count_by_cluster = cat_by_clusters.groupby("assigned_cluster").agg(
        sum=pd.NamedAgg(column="count", aggfunc="sum")).values.reshape(-1)
    # Compute the proportion of sentiment in each cluster
    cat_by_clusters['prop'] = cat_by_clusters.apply(lambda x: x['count'] / count_by_cluster[x['assigned_cluster']],
                                                    axis=1)
    cat_by_clusters, df, df_unique_people = reorder_clusters(cat_by_clusters, df, df_unique_people, k)

    return cat_by_clusters, df, df_unique_people


def count_different_values(assigned_people, col):
    """
    Return the count of each possible value in the list-column given 
    Parameters:
      - assigned_people: The dataframe of all people
      - col: the column we want to count
    """
    return pd.get_dummies(assigned_people[col].apply(pd.Series).stack()) \
        .groupby(level=0) \
        .sum() \
        .sum(axis=0) \
        .to_frame("count")


def print_significancy(type_name, df, nb_people, min_pct, top_five):
    """
    Print the significant values in the asked columns.
    Parameters:
      - type_name: the name of the columns we want to get the values from
      - df: The dataframe with the different values of the column.
      - nb_people: the number of people in the dataframe
      - min_pct: The minimal percentage to display the value
      - top_five: A boolean indicating whether we should print the top_five values in the given column irrespective of if they appears min_pct of time
    """
    count = df[df['count'] / nb_people > min_pct]
    if top_five:
        print(f"The significant {type_name} are:")
        df.sort_values("count", ascending=False).head(5).apply(lambda x: print(f"\t{x.name} with {x['count']} people"),
                                                               axis=1)
        print()
    elif len(count) > 0:
        print(f"The significant {type_name} are:")
        count.apply(lambda x: print(f"\t{x.name} with {x['count']} people"), axis=1)
        print()
    else:
        print(f"No significant {type_name} found")
        print()


def extract_people_from_cluster(df, cluster_id, min_pct=0.3, top_five=False):
    """
    Extract significant value from cluster
    Parameters: 
      - df: the dataframe with all the people
      - cluster_id: The cluster we want to investigate
      - min_pct: The minimal percentage to display the value
      - top_five: A boolean indicating whether we should print the top_five values in the given column irrespective of if they appears min_pct of time

    """
    # Extract people from the requested clsuter
    assigned_people = df[df['assigned_cluster'] == cluster_id]
    nb_people = len(assigned_people)

    # Investigate the different value
    print(f"Number of people in cluster {cluster_id} is {nb_people}.")
    natio_count = count_different_values(assigned_people, "speaker_nationality")
    print_significancy("nationality", natio_count, nb_people, min_pct, top_five)

    cont_count = count_different_values(assigned_people, "speaker_continent")
    print_significancy("continent", cont_count, nb_people, min_pct, False)

    cont_count = count_different_values(assigned_people, "speaker_gender")
    print_significancy("gender", cont_count, nb_people, min_pct, False)

    occ_count = count_different_values(assigned_people, "speaker_occupation")
    print_significancy("occupation", occ_count, nb_people, min_pct, top_five)


"""
Grouping visualisation
"""


def analyse_split(dfs, cat, cat_name, att_name_title, use_plotly=False):
    """
    Perform two-sided t-test between any pairs of dataframe (i.e categories) in the given list of dataframe
    Parameters:
        dfs: The list of dataframes to compare (each dataframe correspond to one category)
        cat: The list of names of all categories.
        cat_name: The name of the category
        att_name_title: Attribute name to put in the title
        use_plotly: if we should use plotly instead of sns or not
    """
    # Compute all the pairs of ranges
    all_pairs = itertools.combinations(list(range(len(dfs))), 2)
    for p1, p2 in all_pairs:
        df1 = dfs[p1]
        df2 = dfs[p2]
        # Compute the t-test and print the if significant
        p_value, alpha, sign = statistical_difference(df1['compound'], df2['compound'])
        if sign:
            print(f"The mean compound score for category " + cat[p1] + f" is {df1['compound'].mean():.4f}")
            print(f"The mean compound score for category " + cat[p2] + f" is {df2['compound'].mean():.4f}")
            print_statistical_test_result(p_value, alpha, sign)
            print()

    # Concatenate all dataframes and plot a boxplot
    all_df = pd.concat(dfs)
    if use_plotly:
        plot_box_plot_mean_compound(all_df, 'cat', cat, att_name_title)
    else:
        ax = sns.boxplot(x="cat", y="compound", data=all_df)
        ax.set_xlabel(cat_name)
        ax.set_ylabel("Mean compound score")
        ax.set_title(f"Boxplot of mean compound score by {att_name_title}")


def analyse_split_on_ages(df, tuple_list=None, use_plotly=False):
    """
    Analyse the given split on ages by doing ttest and box plots
    Parameters:
      - df : The dataframe containing all the quotes
      - tuple_list: A list of tuple containing tuples of upper and lower limit
      use_plotly: if we should use plotly instead of sns or not
    """
    # Remove people that have an age over 125
    if tuple_list is None:
        tuple_list = [(0, 25), (26, 50), (51, 75), (75, 120)]

    dfs = []
    cat = []
    # Split the given dataframe into the given range of ages
    for (lim_inf, lim_sup) in tuple_list:
        cur_df = df.loc[(df['speaker_age'].astype(int) >= lim_inf) & (df['speaker_age'].astype(int) <= lim_sup)].copy()
        cur_df['cat'] = f"{lim_inf}-{lim_sup}"
        cat.append(f"{lim_inf}-{lim_sup}")
        dfs.append(cur_df)

    analyse_split(dfs, cat, 'Age range', "Ages", use_plotly)


def analyse_split_on_gender(df, list_val, use_plotly=False):
    """
    Analyse the given split on gender by doing ttest and box plots
    Parameters:
      - df : The dataframe containing all the quotes
      - list_val: the list of gender we want to investigate
      - use_plotly: if we should use plotly instead of sns or not
    """
    dfs = []
    cat = []
    # Split the given dataframe into the given gender
    for val in list_val:
        cur_df = df[
            df.apply(lambda x: (x['speaker_gender'] is not None) and (val in x['speaker_gender']), axis=1)].copy()
        cur_df['cat'] = val
        cat.append(val)
        dfs.append(cur_df)
    print(cat)

    analyse_split(dfs, cat, 'Gender', "Gender", use_plotly)


def analyse_split_on_continent(df, list_val, use_plotly=False):
    """
    Analyse the given split on continent by doing ttest and box plots
    Parameters:
      - df : The dataframe containing all the quotes
      - list_val: the list of continents we want to investigate
      - use_plotly: if we should use plotly instead of sns or not
    """
    dfs = []
    cat = []
    # Split the given dataframe into the given continent
    for val in list_val:
        cur_df = df[
            df.apply(lambda x: (x['speaker_continent'] is not None) and (val in x['speaker_continent']), axis=1)].copy()
        cur_df['cat'] = val
        cat.append(val)
        dfs.append(cur_df)

    analyse_split(dfs, cat, 'Continent', "Continent", use_plotly)
