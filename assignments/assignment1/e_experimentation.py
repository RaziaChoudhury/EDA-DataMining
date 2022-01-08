import collections
import itertools
from pathlib import Path
from typing import Union, Optional
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import datetime

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.c_data_cleaning import *
from assignments.assignment1.d_data_encoding import *
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################


def process_iris_dataset() -> pd.DataFrame:
    """
    In this example, I call the methods you should have implemented in the other files
    to read and preprocess the iris dataset. This dataset is simple, and only has 5 columns:
    four numeric and one categorical. Depending on what I want to do in the future, I may want
    to transform these columns in other things (for example, I could transform a numeric column
    into a categorical one by splitting the number into bins, similar to how a histogram creates bins
    to be shown as a bar chart).

    In my case, what I want to do is to *remove missing numbers*, replacing them with valid ones,
    and *delete outliers* rows altogether (I could have decided to do something else, and this decision
    will be on you depending on what you'll do with the data afterwords, e.g. what machine learning
    algorithm you'll use). I will also standardize the numeric columns, create a new column with the average
    distance between the three numeric column and convert the categorical column to a onehot-encoding format.

    :return: A dataframe with no missing values, no outliers and onehotencoded categorical columns
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)

    for nc in numeric_columns:
        df = fix_outliers(df, nc)
        df = fix_nans(df, nc)
        df.loc[:, nc] = standardize_column(df.loc[:, nc])

    distances = pd.DataFrame()
    for nc_combination in list(itertools.combinations(numeric_columns, 2)):
        distances[str(nc_combination)] = calculate_numeric_distance(df.loc[:, nc_combination[0]],
                                                                    df.loc[:, nc_combination[1]],
                                                                    DistanceMetric.EUCLIDEAN).values
    df['numeric_mean'] = distances.mean(axis=1)

    for cc in categorical_columns:
        ohe = generate_one_hot_encoder(df.loc[:, cc])
        df = replace_with_one_hot_encoder(df, cc, ohe, list(ohe.get_feature_names()))

    return df


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def process_iris_dataset_again() -> pd.DataFrame:
    """
    Consider the example above and once again perform a preprocessing and cleaning of the iris dataset.
    This time, use normalization for the numeric columns and use label_encoder for the categorical column.
    Also, for this example, consider that all petal_widths should be between 0.0 and 1.0, replace the wong_values
    of that column with the mean of that column. Also include a new (binary) column called "large_sepal_lenght"
    saying whether the row's sepal_length is larger (true) or not (false) than 5.0
    :return: A dataframe with the above conditions.
    """
    df = read_dataset(Path('..', '..', 'iris.csv'))
    numeric_columns = get_numeric_columns(df)
    categorical_columns = get_text_categorical_columns(df)

    for nc in numeric_columns:
        df = fix_outliers(df, nc)
        df = fix_nans(df, nc)
        df.loc[:, nc] = normalize_column(df.loc[:, nc])
    # compute distances for all combinations of numeric values #
    distances = pd.DataFrame()
    for nc_combination in list(itertools.combinations(numeric_columns, 2)):
        distances[str(nc_combination)] = calculate_numeric_distance(df.loc[:, nc_combination[0]],
                                                                    df.loc[:, nc_combination[1]],
                                                                    DistanceMetric.EUCLIDEAN).values
    # compute mean of all numeric distances in df #
    df['numeric_mean'] = distances.mean(axis=1)
    df["large_sepal_length"] = (df["sepal_length"] > 5.0).astype(float)
    # encode all catergorical into label encodings #
    for cc in categorical_columns:
        le = generate_label_encoder(df.loc[:, cc])
        df = replace_with_label_encoder(df, cc, le)
    return df


def process_amazon_video_game_dataset():
    """
    Now use the rating_Video_Games dataset following these rules:
    1. The rating has to be between 1.0 and 5.0
    2. Time should be converted from milliseconds to datetime.datetime format
    3. For the future use of this data, I don't care about who voted what, I only want the average rating per product,
        therefore replace the user column by counting how many ratings each product had (which should be a column called count),
        and the average rating (as the "review" column).
    :return: A dataframe with the above conditions. The columns at the end should be: asin,review,time,count
    """
    df = read_dataset(Path('..', '..', 'ratings_Video_Games.csv'))

    #1. df["review"] must be greater than 0 and less than 6: #
    df = fix_numeric_wrong_values(df, column="review",
                                must_be_rule=WrongValueNumericRule.MUST_BE_GREATER_THAN,
                                must_be_rule_optional_parameter=0)
    df = fix_numeric_wrong_values(df, column="review",
                                  must_be_rule=WrongValueNumericRule.MUST_BE_LESS_THAN,
                                  must_be_rule_optional_parameter=6)
    # fix any nans occuring from the fixing of values #
    df = fix_nans(df, column="review")

    #2. convert ms to datetime #
    df["time"] = df["time"].apply(lambda x: datetime.datetime.fromtimestamp(x))

    #3. get average reviews and counts for each product #
    values = df['asin'].value_counts().reset_index()
    values.columns = ["asin", "count"]
    reviews = df.groupby("asin")["review"].mean()
    time = df.groupby("asin")["time"].mean()
    values = values.join(reviews, on="asin")
    values = values.join(time, on="asin")
    # avg time here might not make sense? #
    return values



def process_amazon_video_game_dataset_again():
    """
    Now use the rating_Video_Games dataset following these rules (the third rule changed, and is more open-ended):
    1. The rating has to be between 1.0 and 5.0, drop any rows not following this rule
    2. Time should be converted from milliseconds to datetime.datetime format
    3. For the future use of this data, I just want to know more about the users, therefore show me how many reviews each user has,
        and a statistical analysis of each user (average, median, std, etc..., each as its own row)
    :return: A dataframe with the above conditions.
    """
    df = read_dataset(Path('..', '..', 'ratings_Video_Games.csv'))

    # 1. df["review"] must be greater than 0 and less than 6: #
    df = fix_numeric_wrong_values(df, column="review",
                                  must_be_rule=WrongValueNumericRule.MUST_BE_GREATER_THAN,
                                  must_be_rule_optional_parameter=0)
    df = fix_numeric_wrong_values(df, column="review",
                                  must_be_rule=WrongValueNumericRule.MUST_BE_LESS_THAN,
                                  must_be_rule_optional_parameter=6)
    # fix any nans occuring from the fixing of values #
    df = fix_nans(df, column="review")

    # 2. convert ms to datetime #
    df["time"] = df["time"].apply(lambda x: datetime.datetime.fromtimestamp(x))

    # 3. group by users #
    user_df = pd.DataFrame()
    users = df.groupby("user")["review"]
    user_df["user"] = df["user"].unique()
    user_stats = {
        "review_mean":users.mean(),
        "review_median":users.median(),
        "review_min":users.min(),
        "review_max":users.max(),
        "review_count":users.count()
    }
    # for users that only have one value, std dev will be set to zero #
    # stats are not very insightful for these cases #
    user_stats["review_std"] = users.std()
    user_stats["review_std"][user_stats["review_std"].isna()] = 0.0
    for label, value in user_stats.items():
        value  = value.reset_index()
        value.columns = ["user", label]
        user_df = user_df.merge(value, on="user")
    return user_df







def process_life_expectancy_dataset():
    """
    Now use the life_expectancy_years and geography datasets following these rules:
    1. The life expectancy dataset has missing values and outliers. Fix them.
    2. The geography dataset has problems with unicode letters. Make sure your code is handling it properly.
    3. Change the format of life expectancy, so that instead of one row with all 28 years, the data has 28 rows, one for each year,
        and with a column "year" with the year and a column "value" with the original value
    4. Merge (or more specifically, join) the two datasets with the common column being the country name (be careful with wrong values here)
    5. Drop all columns except country, continent, year, value and latitude (in this hypothetical example, we wish to analyse differences
        between southern and northern hemisphere)
    6. Change the latitude column from numerical to categorical (north vs south) and pass it though a label_encoder
    7. Change the continent column to a one_hot_encoder version of it
    :return: A dataframe with the above conditions.
    """
    life_e = read_dataset(Path('..', '..', 'life_expectancy_years.csv'))
    geo = read_dataset(Path('..', '..', 'geography.csv'))

    # 1. fix outliers #
    le_numeric_columns = get_numeric_columns(life_e)
    for nc in le_numeric_columns:
        life_e = fix_outliers(life_e, nc)
        life_e = fix_nans(life_e, nc)


    #2. inspect for issues with unicode #
    for type in geo.apply(lambda x: pd.api.types.infer_dtype(x.values)):
        if type == "unicode":
            print("unicode present in dtype, need further inspection")
    """
    running on windows with pandas==1.3.3    
    No issue with unicode
    """
    # 3. melt dataset #
    life_e = life_e.melt(id_vars=['country'])
    life_e.columns = ['name', 'year', 'value']
    # 4. merge datasets on "name" #
    total = life_e.merge(geo, on="name")

    # 5. drop everything but country, continent, year, value and latitude #
    # ASSUMPTION 'four_regions' is the continent attribute
    total = total[['name', 'four_regions', 'year', 'value', 'Latitude']]
    # change name column names #
    total.columns = ['country', 'continent', 'year', 'value', 'Latitude']

    # 6. north, latitude >= 0 else south
    total["Latitude"] = total["Latitude"].apply(lambda x: "north" if x >= 0 else "south")
    # apply label encoder to the column
    le_lat = generate_label_encoder(total.loc[:, "Latitude"])
    total = replace_with_label_encoder(total, 'Latitude', le_lat)

    le_country = generate_label_encoder(total.loc[:, "country"])
    total = replace_with_label_encoder(total, 'country', le_country)

    le_cont = generate_label_encoder(total.loc[:, "continent"])
    total = replace_with_label_encoder(total, "continent", le_cont)
    # 7. one hot encoding for continent #
    # ohe = generate_one_hot_encoder(total.loc[:, 'continent'])
    # total = replace_with_one_hot_encoder(total, 'continent', ohe, list(ohe.get_feature_names()))
    # normalize data
    total["year"] = total["year"].astype(float)
    for col in ['country', 'year', 'value']:
        total[col] = normalize_column(total[col])
    return total

def process_life_expectancy_dataset_lon():
    """
    Now use the life_expectancy_years and geography datasets following these rules:
    1. The life expectancy dataset has missing values and outliers. Fix them.
    2. The geography dataset has problems with unicode letters. Make sure your code is handling it properly.
    3. Change the format of life expectancy, so that instead of one row with all 28 years, the data has 28 rows, one for each year,
        and with a column "year" with the year and a column "value" with the original value
    4. Merge (or more specifically, join) the two datasets with the common column being the country name (be careful with wrong values here)
    5. Drop all columns except country, continent, year, value and latitude (in this hypothetical example, we wish to analyse differences
        between southern and northern hemisphere)
    6. Change the latitude column from numerical to categorical (north vs south) and pass it though a label_encoder
    7. Change the continent column to a one_hot_encoder version of it
    :return: A dataframe with the above conditions.
    """
    life_e = read_dataset(Path('..', '..', 'life_expectancy_years.csv'))
    geo = read_dataset(Path('..', '..', 'geography.csv'))

    # 1. fix outliers #
    le_numeric_columns = get_numeric_columns(life_e)
    for nc in le_numeric_columns:
        life_e = fix_outliers(life_e, nc)
        life_e = fix_nans(life_e, nc)


    #2. inspect for issues with unicode #
    for type in geo.apply(lambda x: pd.api.types.infer_dtype(x.values)):
        if type == "unicode":
            print("unicode present in dtype, need further inspection")
    """
    running on windows with pandas==1.3.3    
    No issue with unicode
    """
    # 3. melt dataset #
    life_e = life_e.melt(id_vars=['country'])
    life_e.columns = ['name', 'year', 'value']
    # 4. merge datasets on "name" #
    total = life_e.merge(geo, on="name")

    # 5. drop everything but country, continent, year, value and latitude #
    # ASSUMPTION 'four_regions' is the continent attribute
    total = total[['name', 'four_regions', 'year', 'value', 'Latitude', 'Longitude']]
    # change name column names #
    total.columns = ['country', 'continent', 'year', 'value', 'Latitude', 'Longitude']

    # apply label encoder to the column
    le_lat = generate_label_encoder(total.loc[:, "Latitude"])
    total = replace_with_label_encoder(total, 'Latitude', le_lat)

    # le_country = generate_label_encoder(total.loc[:, "country"])
    # total = replace_with_label_encoder(total, 'country', le_country)

    # le_cont = generate_label_encoder(total.loc[:, "continent"])
    # total = replace_with_label_encoder(total, "continent", le_cont)
    # 7. one hot encoding for continent #
    # ohe = generate_one_hot_encoder(total.loc[:, 'continent'])
    # total = replace_with_one_hot_encoder(total, 'continent', ohe, list(ohe.get_feature_names()))
    # normalize data
    total["year"] = total["year"].astype(float)
    for col in ['value']:
        total[col] = normalize_column(total[col])
    return total

def process_life_expectancy_dataset_unnorm():
    """
    Now use the life_expectancy_years and geography datasets following these rules:
    1. The life expectancy dataset has missing values and outliers. Fix them.
    2. The geography dataset has problems with unicode letters. Make sure your code is handling it properly.
    3. Change the format of life expectancy, so that instead of one row with all 28 years, the data has 28 rows, one for each year,
        and with a column "year" with the year and a column "value" with the original value
    4. Merge (or more specifically, join) the two datasets with the common column being the country name (be careful with wrong values here)
    5. Drop all columns except country, continent, year, value and latitude (in this hypothetical example, we wish to analyse differences
        between southern and northern hemisphere)
    6. Change the latitude column from numerical to categorical (north vs south) and pass it though a label_encoder
    7. Change the continent column to a one_hot_encoder version of it
    :return: A dataframe with the above conditions.
    """
    life_e = read_dataset(Path('..', '..', 'life_expectancy_years.csv'))
    geo = read_dataset(Path('..', '..', 'geography.csv'))

    # 1. fix outliers #
    le_numeric_columns = get_numeric_columns(life_e)
    for nc in le_numeric_columns:
        life_e = fix_outliers(life_e, nc)
        life_e = fix_nans(life_e, nc)


    #2. inspect for issues with unicode #
    for type in geo.apply(lambda x: pd.api.types.infer_dtype(x.values)):
        if type == "unicode":
            print("unicode present in dtype, need further inspection")
    """
    running on windows with pandas==1.3.3    
    No issue with unicode
    """
    # 3. melt dataset #
    life_e = life_e.melt(id_vars=['country'])
    life_e.columns = ['name', 'year', 'value']
    # 4. merge datasets on "name" #
    total = life_e.merge(geo, on="name")

    # 5. drop everything but country, continent, year, value and latitude #
    # ASSUMPTION 'four_regions' is the continent attribute
    total = total[['name', 'four_regions', 'year', 'value', 'Latitude', 'Longitude']]
    # change name column names #
    total.columns = ['country', 'continent', 'year', 'value', 'Latitude', 'Longitude']

    # apply label encoder to the column
    le_lat = generate_label_encoder(total.loc[:, "Latitude"])
    total = replace_with_label_encoder(total, 'Latitude', le_lat)

    # le_country = generate_label_encoder(total.loc[:, "country"])
    # total = replace_with_label_encoder(total, 'country', le_country)

    # le_cont = generate_label_encoder(total.loc[:, "continent"])
    # total = replace_with_label_encoder(total, "continent", le_cont)
    # 7. one hot encoding for continent #
    # ohe = generate_one_hot_encoder(total.loc[:, 'continent'])
    # total = replace_with_one_hot_encoder(total, 'continent', ohe, list(ohe.get_feature_names()))
    # normalize data
    total["year"] = total["year"].astype(float)
    #for col in ['value']:
    #    total[col] = normalize_column(total[col])
    return total

if __name__ == "__main__":
    # assert process_iris_dataset() is not None
    # assert process_iris_dataset_again() is not None
    # assert process_amazon_video_game_dataset() is not None
    # assert process_amazon_video_game_dataset_again() is not None
    assert process_life_expectancy_dataset() is not None
