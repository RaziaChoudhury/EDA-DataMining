from assignments.assignment2.imports import *
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS, DBSCAN, AffinityPropagation, SpectralClustering
from sklearn import metrics

"""
Clustering is an unsupervised form of machine learning. It uses unlabeled data and returns the similarity/dissimilarity between rows of the data.
See https://scikit-learn.org/stable/modules/clustering.html for an overview of methods in sklearn.
"""


#######################################################
# Read the comments carefully in the following method(s)
# Assignment questions are marked as "??subtask"
#######################################################


def simple_k_means(x: pd.DataFrame, n_clusters=3, score_metric='euclidean') -> Dict:
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_transform(x)

    # There are many methods of deciding a score of a cluster model. Here is one example:
    score = metrics.silhouette_score(x, model.labels_, metric=score_metric)
    return dict(model=model, score=score, clusters=clusters)

def simple_k_means_pred(x: pd.DataFrame, n_clusters=3, score_metric='euclidean') -> Dict:
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_predict(x)

    # There are many methods of deciding a score of a cluster model. Here is one example:
    score = metrics.silhouette_score(x, model.labels_, metric=score_metric)
    return dict(model=model, score=score, clusters=clusters)


def iris_clusters() -> Dict:
    """
    Let's use the iris dataset and clusterise it:
    """
    iris = process_iris_dataset_again()

    iris.drop("large_sepal_length", axis=1, inplace=True)

    # Let's generate the clusters considering only the numeric columns first
    no_species_column = simple_k_means(iris.iloc[:, :4])

    ohe = generate_one_hot_encoder(iris['species'])
    df_ohe = replace_with_one_hot_encoder(iris.copy(deep=False), 'species', ohe, list(ohe.get_feature_names()))

    # Notice that here I have binary columns, but I am using euclidean distance to do the clustering AND score evaluation
    # This is pretty bad
    no_binary_distance_clusters = simple_k_means(df_ohe)

    # Finally, lets use just a label encoder for the species.
    # It is still bad to change the labels to numbers directly because the distances between them does not make sense
    le = generate_label_encoder(iris['species'])
    df_le = replace_with_label_encoder(iris.copy(deep=False), 'species', le)
    labeled_encoded_clusters = simple_k_means(df_le)

    # See the result for yourself:
    r1 = round(no_species_column['score'], 2)
    r2 = round(no_binary_distance_clusters['score'], 2)
    r3 = round(labeled_encoded_clusters['score'], 2)
    print(
        f"Clustering Scores:\nno_species_column:{r1}, no_binary_distance_clusters:{r2}, labeled_encoded_clusters:{r3}")

    return max(r1, r2, r3)


##############################################
# Implement all the below methods
# Don't install any other python package other than provided by python or in requirements.txt
##############################################
def custom_clustering(x: pd.DataFrame) -> Dict:
    """
    As you saw before, it is much harder to apply the right distance metrics. Take a look at:
    https://scikit-learn.org/stable/modules/clustering.html
    and check the metric used for each implementation. You will notice that suppositions were made,
    which makes harder to apply these clustering algorithms as-is due to the metrics used.
    Also go into each and notice that some of them there is a way to choose a distance/similarity/affinity metric.
    You don't need to check how each technique is implemented (code/math), but do use the information from the clustering
    lecture and check the parameters of the method (especially if there is any distance metric available among them).
    Chose one of them which is, in your opinion, generic enough, and justify your choice with a comment in the code (1 sentence).
    The return of this method should be the model, a score (e.g. silhouette
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) and the result of clustering the
    input dataset.

    """

    # DBSCAN is chosen, it is effective with outliers and is scalable for large numbers of samples and medium classes and
    # has only one parameter to set and does not require to predetermine number of clusters making it more general purpose,
    # OPTICS found to be too expensive for compute constraints, spectral clustering, agglomerative clustering found to be
    # too problem specific (too many hparams), and affinity propogation not scalable with samples therefore not a good choice.
    # hyper param selection eps and min samples both are associated with the density
    # eps parameter adjusted to 0.3 to maximize results in three test cases, discovered through trial and error,
    # cosine distance was experimented with, but found it was very sensitive to the epsilon for each individual case,
    # this is because it is related to the proportionality of vectors rather than distance
    dbscan = DBSCAN(eps=0.3, metric='euclidean').fit(x)
    # dbscan = DBSCAN(eps=0.002, metric='cosine').fit(x)
    score = metrics.silhouette_score(x, dbscan.labels_, metric='euclidean')
    clusters = dbscan.fit_predict(x)

    return dict(model=dbscan, score=score, clusters=clusters)



def cluster_iris_dataset_again() -> Dict:
    """
    Run the df returned form process_iris_dataset_again() method of A1 e_experimentation file through the custom_clustering and discuss (3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    """
    RESULTS: 
    Number of clusters: 3,
    Score: 0.8608343159908078
    The models sillohuette score has no improvement on the k-means algorithm, and it discovers three clusters, which is what 
    was used for the k-means experiment. One fault in this is the use of the species variable, the distance of labels is 
    undefined and therefore the approach is faulted. It is notable that when using cosine distance metric and one-hot encodings, 
    a significant improvement is (Number of clusters = 3, silloheutte score = 0.99121), this is due to the functionality of the cosine distance metric and its consideration of 
    the vector's direction rather than distance between points, however for the purpose of keeping the algorithm general the euclidean distance was used 
    with label encodings instead. 
    """
    # drop sepal length because it is correlated to the length and therefore not useful #
    iris = process_iris_dataset_again().drop(columns=["large_sepal_length"])
    model = custom_clustering(iris)
    print(f"Number of clusters: {len(set(model['clusters']))},\nScore: {model['score']}")
    return dict(model=model, score=model['score'], clusters=model['clusters'])


def cluster_amazon_video_game() -> Dict:
    """
    Run the df returned from process_amazon_video_game_dataset() methos of A1 e_experimentation file through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    """
    Results:
    Number of clusters: 243,
    Score: 0.6217476549036823
        
    The sillohuette score indicates that the clusters are not well seperated but are more correctly labelled, the number of clusters is 243. 
    When considering that the DBSCAN is good with determining outliers, 
    and also that the data consists of two important fields: number of reviews and 5 star rating, it is clear that there would be significant overlap in 
    whichever classes would be formed, therefore a sillohuette score of 0.62 speaks to this, that is there is a little bit of 
    cluster seperation, but not fully seperated. The data is not rich with possible clusterings, it is large and simple leading to a relatively large number of 
    clusters and sub-par sillohuette score when thinking about what the clusters could represent, there isnt many interesting 
    thoughts because it is such a simple dataset. 
    """
    amazon1 = process_amazon_video_game_dataset()
    # the asin and time columns are unique identifers and therefore should be dropped #
    amazon1 = amazon1.drop(columns=["asin", "time"])

    model = custom_clustering(amazon1)
    print(f"Number of clusters: {len(set(model['clusters']))},\nScore: {model['score']}")
    return dict(model=model, score=model['score'], clusters=model['clusters'])


def cluster_amazon_video_game_again() -> Dict:
    """
    Run the df returned from process_amazon_video_game_dataset_again() methos of A1 e_experimentation file through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """
    """
    RESULTS:
    Number of clusters: 162,
    Score: 0.9714079854454799
    
    The high sillohuette score indicates that the 162 formed clusters are very well seperated. When considering the data
    it makes sense that this is possible, one possible clustering could be the different types of reviewers being clustered.
    Because the entries are all numerical, the euclidean metric makes sense as well. It is possible that there are 162 clear
    divisions in the users in amazon, that tend to review in certain ways, e.g. some users may be more negative whereas others
    more positive and so on.
    
    """
    # load data and drop user column because it is unique #
    amazon2 = process_amazon_video_game_dataset_again().drop(columns=["user"])
    # because dataset is so large we will randomly sample the dataset: #
    amazon2 = amazon2.sample(frac=0.05, replace=False, random_state=42)
    model = custom_clustering(amazon2)

    print(f"Number of clusters: {len(set(model['clusters']))},\nScore: {model['score']}")
    return dict(model=model, score=model['score'], clusters=model['clusters'])


def cluster_life_expectancy() -> Dict:
    """
    Run the df returned from process_amazon_video_game_dataset_again() methos of A1 e_experimentation file through the custom_clustering and discuss (max 3 sentences)
    the result (clusters and score) and also say any limitations (e.g. problems with metrics) that you find.
    We are not looking for an exact answer, we want to know if you really understand your choice and the results of custom_clustering.
    Once again, don't worry about the clustering technique implementation, but do analyse the data/result and check if the clusters makes sense.
    """


    """
    RESULTS:
    Number of clusters: 8,
    Score: 0.6165425894848212
    
    An interesting part of this result is that there are 8 clusters, it is possible that the clusters are formed based on 
    different geographic locations. A clearm problem here is the continent and year columns, these may not logically make sense
    in euclidean space, therefore may be a fault point in the algorithm. The sillohuette score of 0.61 is not overly strong, 
    therefore the clusters are not strongly seperated, but still it indicates that labels are properly assigned, 
    and there exists overlap with the clusters. 
    """
    # note that in the e_experimentation.py file, the data was modified to have
    # continent as a label encoding rather than OHE
    life_e = process_life_expectancy_dataset()
    model = custom_clustering(life_e)

    print(f"Number of clusters: {len(set(model['clusters']))},\nScore: {model['score']}")
    return dict(model=model, score=model['score'], clusters=model['clusters'])


def run_clustering():
    start = time.time()
    print("Clustering in progress...")
    assert iris_clusters() is not None
    assert len(cluster_iris_dataset_again().keys()) == 3
    assert len(cluster_amazon_video_game().keys()) == 3
    assert len(cluster_amazon_video_game_again().keys()) == 3
    assert len(cluster_life_expectancy().keys()) == 3

    end = time.time()
    run_time = round(end - start, 4)
    print("Clustering ended...")
    print(f"{30 * '-'}\nClustering run_time:{run_time}s\n{30 * '-'}\n")


if __name__ == "__main__":
    run_clustering()
