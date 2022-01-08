from imports import *
from pandas.api.types import is_numeric_dtype
from functools import partial
import numpy as np

import a_classification
import b_regression
import c_clustering
"""
The below method should:
?? subtask1  Handle any dataset (if you think worthwhile, you should do some pre-processing)
?? subtask2  Generate a (classification, regression or clustering) model based on the label_column 
             and return the one with best score/accuracy

The label_column can be categorical, numerical or None
-If categorical, run through ML classifiers in "a_classification" file and return the one with highest accuracy: 
    DecisionTree, RandomForestClassifier, KNeighborsClassifier or NaiveBayes
-If numerical, run through these ML regressors in "b_regression" file and return the one with least MSE error: 
    svm_regressor_1(), random_forest_regressor_1()
-If None, run through simple_k_means() and custom_clustering() and return the one with highest silhouette score.
(https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
"""


def generate_model(df: pd.DataFrame, label_column: Optional[pd.Series] = None) -> Dict:
    # model_name is the type of task that you are performing.
    # Use sensible names for model_name so that we can understand which ML models if executed for given df.
    # ex: Classification_random_forest or Regression_SVM.
    # model is trained model from ML process
    # final_score will be Accuracy in case of Classification, MSE in case of Regression and silhouette score in case of clustering.
    # your code here.
    # ?? subtask1  Handle any dataset (if you think worthwhile, you should do some pre-processing)
    # data preprocessing #

    # clean numeric columns #
    numeric_columns = get_numeric_columns(df)
    for nc in numeric_columns:
        df = fix_outliers(df, nc)
        # df = fix_nans(df, nc)


    if label_column is not None:
        # for both regression and classification perform standardization #
        for nc in numeric_columns:
            df[nc] = normalize_column(df[nc])

        # drop nans
        df = df[label_column.notna()]
        label_column = label_column[label_column.notna()]
        df = df.dropna(axis=1)
        label_column = label_column[df.index]



        # infer dtype #
        if is_numeric_dtype(label_column):
            # numeric label, therfore regression problem
            model_type = "regression"
            # mse to measure performance #
            metric = "mse"
            # comparison function, want minimal mse #
            compare = lambda x, y: x < y
            # assign models to test #
            models = [(partial(b_regression.svm_regressor_1,y=label_column), "svm_regressor"),
                      (partial(b_regression.random_forest_regressor_1,y=label_column), "random_forest_regressor"),
                      ]
        else:
            # if not numeric, classification
            model_type = "classification"
            # acc to measure performance (not a thorough choice)#
            metric = "accuracy"
            # comparison function, want maximal acc #
            compare = lambda x, y: x > y
            # label encode the targets #
            le = generate_label_encoder(label_column)
            label_column = pd.DataFrame(le.transform(label_column.to_numpy()))
            # assign models to test #
            models = [(partial(a_classification.decision_tree_classifier,y=label_column), "decision_tree_classifier"),
                       (partial(a_classification.random_forest_classifier,y=label_column), "random_forest_classifier"),
                      (partial(a_classification.knn_classifier,y=label_column), "knn_classifier"),
                      (partial(a_classification.naive_bayes_classifier,y=label_column), "naive_bayes_classifier")]

    else:
        # if no label, cluster #
        model_type = "clustering"
        # metric to measure performance #
        metric = "score"
        # comparison function, want maximal score #
        compare = lambda x, y: x > y
        models = [(c_clustering.simple_k_means, "k_means_cluster"),
                  (c_clustering.custom_clustering, "custom_cluster")]

    # ?? subtask2  Generate a (classification, regression or clustering) model based on the label_column and return the one with best score/accuracy
    # run all models #
    max_score = np.inf if model_type == "regression" else -np.inf
    max_name = ""
    for model, name in models:
        model = model(df)
        score = model[metric]
        # compare best #
        if compare(score, max_score):
            max_score = score
            max_name = name
            max_model = model


    return dict(model_name=max_name, model=max_model["model"], final_score=max_score)


def run_custom():
    start = time.time()
    print("Custom modeling in progress...")
    df = pd.DataFrame()  # Markers will run your code with a separate dataset unknown to you.

    result = generate_model(df)
    print(f"result:\n{result}\n")

    end = time.time()
    run_time = round(end - start)
    print("Custom modeling ended...")
    print(f"{30 * '-'}\nCustom run_time:{run_time}s\n{30 * '-'}\n")


if __name__ == "__main__":
    run_custom()
