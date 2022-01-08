import collections
from pathlib import Path
from typing import Union, Optional
from enum import Enum
import math
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.a_load_file import read_dataset


##############################################
# Example(s). Read the comments in the following method(s)
##############################################
class WrongValueNumericRule(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    MUST_BE_POSITIVE = 0
    MUST_BE_NEGATIVE = 1
    MUST_BE_GREATER_THAN = 2
    MUST_BE_LESS_THAN = 3


class DistanceMetric(Enum):
    """
    You'll use these enumeration possibilities in your implemented methods below
    """
    EUCLIDEAN = 0
    MANHATTAN = 1


##############################################
# Implement all the below methods
# All methods should be dataset-independent, using only the methods done in the assignment
# so far and pandas/numpy/sklearn for the operations
##############################################
def fix_numeric_wrong_values(df: pd.DataFrame,
                             column: str,
                             must_be_rule: WrongValueNumericRule,
                             must_be_rule_optional_parameter: Optional[float] = None) -> pd.DataFrame:
    """
    This method should fix the wrong_values depending on the logic you think best to do and using the rule passed by parameter.
    Remember that wrong values are values that are in the dataset, but are wrongly inputted (for example, a negative age).
    Here you should only fix them (and not find them) by replacing them with np.nan ("not a number", or also called "missing value")
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :param must_be_rule: one of WrongValueNumericRule identifying what rule should be followed to flag a value as a wrong value
    :param must_be_rule_optional_parameter: optional parameter for the "greater than" or "less than" cases
    :return: The dataset with fixed column
    """
    # store all possible outcomes as dict # (switch)
    rule_functions = {
                     0: lambda x: x if x >= 0 else np.nan,
                     1: lambda x: x if x < 0 else np.nan,
                     2: lambda x: x if x > must_be_rule_optional_parameter else np.nan,
                     3: lambda x : x if x < must_be_rule_optional_parameter else np.nan
                 }
    # apply the appropriate function to the df col #
    df[column] = df[column].apply(rule_functions[must_be_rule.value])
    return df


def fix_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix the column in respective to outliers depending on the logic you think best to do.
    Feel free to choose which logic you prefer, but if you are in doubt, use the simplest one to remove the row
    of the dataframe which is an outlier (note that the issue with this approach is when the dataset is small,
    dropping rows will make it even smaller).
    Remember that some datasets are large, and some are small, so think wisely on how to calculate outliers
    and when to remove/replace them. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The dataset with fixed column
    """

    if column in get_numeric_columns(df):
        # determine outlier based on z-score
        mean = get_column_mean(df, column)
        std = np.std(df[column])
        # if z score > 3 replace with mean
        replace_w_mean = lambda x: mean if (x - mean) / std > 3 else x
        # replace numerical values with their mean if they have z score > 3
        df[column] = df[column].apply(replace_w_mean)

    elif column in get_binary_columns(df):
        # drop rows that do not contain boolean value,
        # if a bool is true or false it will not be anomolous
        df = df[df[column].map(type) == bool]

    elif column in get_text_categorical_columns(df):
        # if doesnt contain string, remove...
        df = df[df[column].map(type) == str]

    elif df[column].dtype == np.datetime64:
        # compute outliers based on z-score for date time
        mean = get_column_mean(df, column)
        std = np.std(df[column])
        # if z score > 3 replace with mean
        replace_w_mean = lambda x: mean if (x - mean) / std > 3 else x
        df[column] = df[column].apply(replace_w_mean)

    return df




def fix_nans(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    This method should fix all nans (missing data) depending on the logic you think best to do
    Remember that some datasets are large, and some are small, so think wisely on when to use each possible
    removal/fix/replace of nans. Also remember that some columns are numeric, some are categorical, some are binary
    and some are datetime. Use the methods in b_dataset_profile to your advantage!
    :param df: Dataset
    :param column: the column to be investigated and fixed
    :return: The fixed dataset
    """
    if column in get_numeric_columns(df) or df[column].dtype == np.datetime64:
        if len(df[df[column].isna()]) / len(df) < 0.1: # if less than 10 percent of data is NaN, just drop it
            df = df[df[column].notna()]
        else:
            # take mean
            mean = df[column].mean()
            # add mean
            df[column] = df[column].apply(lambda x: mean if math.isnan(x) else x)



    elif column in get_binary_columns(df) or get_text_categorical_columns(df):
        # drop row for all binary missed vales
        df = df[df[column].notna()]
    return df


def normalize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and normalise it between 0 and 1.
    :param df_column: Dataset's column
    :return: The column normalized
    """
    # convert all values to positive values
    min = df_column.min()
    df_column = df_column.apply(lambda x: x + abs(min))
    max_val = df_column.max()
    df_column = df_column.apply(lambda x: x/max_val)
    return df_column

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    apply standard scaler to entire dataset
    :param df:
    :return:
    """
    sc = StandardScaler()
    return pd.DataFrame(sc.fit_transform(df), columns=df.columns)


def standardize_column(df_column: pd.Series) -> pd.Series:
    """
    This method should recalculate all values of a numeric column and standardize it between -1 and 1 with its average at 0.
    :param df_column: Dataset's column
    :return: The column standardized
    """
    # compute the value based on adjusted min max formula #
    max_val = df_column.max()
    min_val = df_column.min()
    df_column.apply(lambda x: 2*((x - min_val)/(max_val-min_val)) - 1)
    return df_column



def calculate_numeric_distance(df_column_1: pd.Series, df_column_2: pd.Series, distance_metric: DistanceMetric) -> pd.Series:
    """
    This method should calculate the distance between two numeric columns
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :param distance_metric: One of DistanceMetric, and for each one you should implement its logic
    :return: A new 'column' with the distance between the two inputted columns
    """
    if distance_metric.value: # Manhattan
        return pd.Series(abs(df_column_1 - df_column_2))
    else:# euclidean
        return pd.Series(np.sqrt((df_column_1 - df_column_2) ** 2))




from scipy.spatial import distance
def calculate_binary_distance(df_column_1: pd.Series, df_column_2: pd.Series) -> pd.Series:
    """
    This method should calculate the distance between two binary columns.
    Choose one of the possibilities shown in class for future experimentation.
    :param df_column_1: Dataset's column
    :param df_column_2: Dataset's column
    :return: A new 'column' with the distance between the two inputted columns
    """
    # compute hamming distance for each value in column
    ham = lambda x: distance.hamming(x[0], x[1])
    df = pd.concat([df_column_1, df_column_2], axis=1).apply(ham, axis=1)
    return df

if __name__ == "__main__":
    df = pd.DataFrame({'a':[1,2,3,None], 'b': [True, True, False, None], 'c': ['one', 'two', np.nan, None]})
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_LESS_THAN, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_GREATER_THAN, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_POSITIVE, 2) is not None
    assert fix_numeric_wrong_values(df, 'a', WrongValueNumericRule.MUST_BE_NEGATIVE, 2) is not None
    assert fix_outliers(df, 'c') is not None
    assert fix_nans(df, 'c') is not None
    assert normalize_column(df.loc[:, 'a']) is not None
    assert standardize_column(df.loc[:, 'a']) is not None
    assert calculate_numeric_distance(df.loc[:, 'a'], df.loc[:, 'a'], DistanceMetric.EUCLIDEAN) is not None
    assert calculate_numeric_distance(df.loc[:, 'a'], df.loc[:, 'a'], DistanceMetric.MANHATTAN) is not None
    assert calculate_binary_distance(df.loc[:, 'b'], df.loc[:, 'b']) is not None
    print("ok")
