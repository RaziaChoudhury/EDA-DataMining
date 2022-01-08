from assignments.assignment2.imports import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.model_selection import train_test_split



"""
Classification is a supervised form of machine learning, which means that it uses labeled data to train a model that 
can predict the category/class of the given input/unseen data.

In this subtask, you'll be training and testing 4 different types of classification models and compare the performance of same for 3 different datasets.
Dataset1: Iris dataset from Assignment1 before normalization.
Dataset2: Iris dataset from Assignment1 after normalization. Notice what change normalization does to the prediction resutls!
Dataset3: Life_expectancy dataset from Assignment1 after label encoding and normalization.
"""


#######################################################
# Read the comments carefully in the following method(s)
# Assignment questions are marked as "??subtask"
#######################################################


# MODEL_1:
def decision_tree_classifier(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a Decision Tree classifier for given dataset. 
    Refer the official documention below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    :param x: data features
    :param y: data labels/class
    :return: trained_model and below given performace_scores for test dataset
    """
    # copying df and series
    x, y = x.copy(deep=False), y.copy(deep=False)
    # ??subtask1: Split the data into 80%-20% train-test sets.
    #             Randomize the data selection.i.e. train and test data shoud be randomly selected in 80/20 ratio.
    #             Use a random_state=42, so that we can recreate same splitting when run multiple times.
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    # ??subtask2: Train your model using train set.
    dt_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    # ??subtask3: Predict test labels/classes for test set.
    y_pred = dt_model.predict(X_test)
    # ??subtask4: Measure the below given performance measures on test predictions.
    dt_confusion_matrix = confusion_matrix(y_test, y_pred)
    # get evaluation report of model #
    dt_accuracy = accuracy_score(y_test, y_pred)
    # taking weighted average for these values, this allows us
    # to get a robust metric that handles any label imbalance
    dt_precision = precision_score(y_test, y_pred, average='weighted')
    dt_recall = recall_score(y_test, y_pred, average='weighted')
    dt_f1_score = f1_score(y_test, y_pred, average='weighted')

    return dict(model=dt_model, confusion_matrix=dt_confusion_matrix, accuracy=dt_accuracy, precision=dt_precision,
                recall=dt_recall, f1_score=dt_f1_score)


def decision_tree_classifier_a3(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a Decision Tree classifier for given dataset.
    Refer the official documention below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    :param x: data features
    :param y: data labels/class
    :return: trained_model and below given performace_scores for test dataset
    """
    # copying df and series
    x, y = x.copy(deep=False), y.copy(deep=False)
    # ??subtask1: Split the data into 80%-20% train-test sets.
    #             Randomize the data selection.i.e. train and test data shoud be randomly selected in 80/20 ratio.
    #             Use a random_state=42, so that we can recreate same splitting when run multiple times.
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    # ??subtask2: Train your model using train set.
    dt_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    # ??subtask3: Predict test labels/classes for test set.
    y_pred = dt_model.predict(X_test)
    return X_test, y_test, y_pred
# MODEL_2:
def random_forest_classifier(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a random forest classifier for given data. 
    Refer the official documention below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    
    In this api, "n_estimator" decides number of trees in the forest, lower value means lower trees and computation
    with the default values of these parameters, the model may take a lot time for computation.
    If necessary, change the "n_estimators", "max_depth" and "max_leaf_nodes" to accelerate the model
    training, but don't forget to comment why you did and any consequences of setting them!
    
    :param x: data features
    :param y: data labels/class
    :return: trained_model and below given performace_scores for test dataset
    """

    # copying df and series
    x, y = x.copy(deep=False), y.copy(deep=False)
    # ??subtask1: Split the data into 80%-20% train-test sets.
    #             Randomize the data selection.i.e. train and test data shoud be randomly selected in 80/20 ratio.
    #             Use a random_state=42, so that we can recreate same splitting when run multiple times.
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # ??subtask2: Train your model using train set.
    rf_model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    # ??subtask3: Predict test labels/classes for test set.
    y_pred = rf_model.predict(X_test)
    # ??subtask4: Measure the below given performance measures on test predictions.
    rf_confusion_matrix = confusion_matrix(y_test, y_pred)
    # get evaluation report of model #
    rf_accuracy = accuracy_score(y_test, y_pred)
    # taking weighted average for these values, this allows us
    # to get a robust metric that handles any label imbalance
    rf_precision = precision_score(y_test, y_pred, average='weighted')
    rf_recall = recall_score(y_test, y_pred, average='weighted')
    rf_f1_score = f1_score(y_test, y_pred, average='weighted')

    return dict(model=rf_model, confusion_matrix=rf_confusion_matrix, accuracy=rf_accuracy, precision=rf_precision,
                recall=rf_recall, f1_score=rf_f1_score)


# MODEL_3:
def knn_classifier(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a K-Nearest Neighbors(KNN) classifier for given data. 
    Refer the official documention below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    
    :param x: data features
    :param y: data labels/class
    :return: trained_model and below given performace_scores for test dataset
    """

    # copying df and series
    x, y = x.copy(deep=False), y.copy(deep=False)
    # ??subtask1: Split the data into 80%-20% train-test sets.
    #             Randomize the data selection.i.e. train and test data shoud be randomly selected in 80/20 ratio.
    #             Use a random_state=42, so that we can recreate same splitting when run multiple times.
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # ??subtask2: Train your model using train set.
    knn_model = KNeighborsClassifier().fit(X_train, y_train)
    # ??subtask3: Predict test labels/classes for test set.
    y_pred = knn_model.predict(X_test)
    # ??subtask4: Measure the below given performance measures on test predictions.
    knn_confusion_matrix = confusion_matrix(y_test, y_pred)
    # get evaluation report of model #
    knn_accuracy = accuracy_score(y_test, y_pred)
    # taking weighted average for these values, this allows us
    # to get a robust metric that handles any label imbalance
    knn_precision = precision_score(y_test, y_pred, average='weighted')
    knn_recall = recall_score(y_test, y_pred, average='weighted')
    knn_f1_score = f1_score(y_test, y_pred, average='weighted')

    return dict(model=knn_model, confusion_matrix=knn_confusion_matrix, accuracy=knn_accuracy, precision=knn_precision,
                recall=knn_recall, f1_score=knn_f1_score)


# MODEL_4:
def naive_bayes_classifier(x: pd.DataFrame, y: pd.Series) -> Dict:
    """
    Method to train and test a Naive Bayes classifier for given data. 
    Refer the official documention below to find api parameters and examples on how to train and test your model.
    https://scikit-learn.org/stable/modules/naive_bayes.html
    
    :param x: data features
    :param y: data labels/class
    :return: trained_model and below given performace_scores for test dataset
    """
    # copying df and series
    x, y = x.copy(deep=False), y.copy(deep=False)
    # ??subtask1: Split the data into 80%-20% train-test sets.
    #             Randomize the data selection.i.e. train and test data shoud be randomly selected in 80/20 ratio.
    #             Use a random_state=42, so that we can recreate same splitting when run multiple times.
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # ??subtask2: Train your model using train set.
    nv_model = GaussianNB().fit(X_train, y_train)
    # ??subtask3: Predict test labels/classes for test set.
    y_pred = nv_model.predict(X_test)
    # ??subtask4: Measure the below given performance measures on test predictions.
    nv_confusion_matrix = confusion_matrix(y_test, y_pred)
    # get evaluation report of model #
    nv_accuracy = accuracy_score(y_test, y_pred)
    # taking weighted average for these values, this allows us
    # to get a robust metric that handles any label imbalance
    nv_precision = precision_score(y_test, y_pred, average='weighted')
    nv_recall = recall_score(y_test, y_pred, average='weighted')
    nv_f1_score = f1_score(y_test, y_pred, average='weighted')

    return dict(model=nv_model, confusion_matrix=nv_confusion_matrix, accuracy=nv_accuracy, precision=nv_precision,
                recall=nv_recall, f1_score=nv_f1_score)


def run_classification_models(x, y):
    """
    This method takes input features and labels and calls all the functions which trains and tests the above 4 Classification models.
    No need to do anything here.
    """
    r1 = decision_tree_classifier(x, y)
    assert len(r1.keys()) == 6
    r2 = random_forest_classifier(x, y)
    assert len(r2.keys()) == 6
    r3 = knn_classifier(x, y)
    assert len(r3.keys()) == 6
    r4 = naive_bayes_classifier(x, y)
    assert len(r4.keys()) == 6

    return r1, r2, r3, r4


def run_classification():
    start = time.time()
    print("Classification in progress...")
    # Dataset1:
    iris = read_dataset(Path('../../iris.csv'))
    label_col = "species"
    feature_cols = iris.columns.tolist()
    feature_cols.remove(label_col)
    x_iris = iris[feature_cols]
    y_iris = iris[label_col]
    result1, result2, result3, result4 = run_classification_models(x_iris, y_iris)

    # Observe all 4 results and notice which model is preforming better.
    print(
        f"{10 * '*'}Dataset1:{iris.shape}{10 * '*'}\nDecision Tree: {result1}\nRandom forest: {result2}\nKNN: {result3}\nNaive Bayes: {result4}\n")

    # Dataset2:
    iris_again = process_iris_dataset_again()
    """
    
    """
    # ??subtask1: process iris_again dataset returned from A1 process_iris_dataset_again() method so that all the categorical columns are label encoded.
    #??subtask2: make sure that all the feature columns are normalized to range [0,1] except label encoded "species" column. These are lables to be predicted Use methods from Assignment1 for this.
    # both sub tasks performed in the e_experimentation.py file
    label_col = "species"
    feature_cols = iris_again.columns.tolist()
    feature_cols.remove(label_col)
    x_iris_again = iris_again[feature_cols]
    y_iris_again = iris_again[label_col]
    result1, result2, result3, result4 = run_classification_models(x_iris_again, y_iris_again)
    # Observe all 4 results and notice which model is preforming better.
    print(
        f"{10 * '*'}Dataset2:{iris_again.shape}{10 * '*'}\nDecision Tree: {result1}\nRandom forest: {result2}\nKNN: {result3}\nNaive Bayes: {result4}\n")

    # Dataset3:
    life_expectancy = process_life_expectancy_dataset()
    """
    ??subtask1: process life_expectancy dataset returned from A1 process_life_expectancy_dataset() method so that all the categorical columns are label encoded.
    # already done in first asn
    """
    # ??subtask2: make sure that all the feature columns are normalized to range [0,1] except label encoded "Latitude" column. These are lables to be predicted. (0/1-North/South)
    #     Use methods from Assignment1 for this.
    le_numeric_columns = get_numeric_columns(life_expectancy)
    for col in  le_numeric_columns:
        life_expectancy[col] = normalize_column(life_expectancy[col])
    label_col = "Latitude"
    feature_cols = life_expectancy.columns.tolist()
    feature_cols.remove(label_col)
    x_life_expectancy = life_expectancy[feature_cols]
    y_life_expectancy = life_expectancy[label_col]

    result1, result2, result3, result4 = run_classification_models(x_life_expectancy, y_life_expectancy)
    # Observe all 4 results and notice which model is preforming better.
    print(
        f"{10 * '*'}Dataset3:{life_expectancy.shape}{10 * '*'}\nDecision Tree: {result1}\nRandom forest: {result2}\nKNN: {result3}\nNaive Bayes: {result4}\n")

    end = time.time()
    run_time = round(end - start, 4)
    print("Classification ended...")
    print(f"{30 * '-'}\nClassification run_time:{run_time}s\n{30 * '-'}\n")


if __name__ == "__main__":
    run_classification()

#Since we are using same parameter names (x: pd.DataFrame, y: pd.Series) in most methods, remember to copy the df passed as parameter 
# and work with the df_copy to avoid warnings and unexpected results. Its a standard practice!
