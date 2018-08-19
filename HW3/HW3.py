#!/usr/bin/python3
#CS7641 HW3 by Tian Mi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_samples, silhouette_score

#################################################
#Data set 1: wine quality data set

data = pd.read_csv('winequality-data.csv')
X = data.iloc[:,:11]
y = data.iloc[:,11]
features = list(X.columns.values)

scaler = MinMaxScaler(feature_range=[0,100])
scaler.fit(X)
X_norm = pd.DataFrame(scaler.transform(X))

#################################################
#K means clustering

range_n_clusters = [2,4,6,8]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X_norm) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X_norm)
    cluster_labels = clusterer.labels_
    print("NMI score: %.6f" % normalized_mutual_info_score(y, cluster_labels))

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X_norm, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_norm, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter( X_norm.iloc[:, 10], X_norm.iloc[:, 8], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_

    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 10], centers[:, 8], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter( c[10], c[8], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()

#################################################
#Expectation Maximization clustering

for n_clusters in range_n_clusters:
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    ax = fig.add_subplot(111)
    
    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = GaussianMixture(n_components=n_clusters, random_state=10).fit(X_norm)
    cluster_labels = clusterer.predict(X_norm)
    print("NMI score: %.6f" % normalized_mutual_info_score(y, cluster_labels))

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    plt.scatter( X_norm.iloc[:, 10], X_norm.iloc[:, 8], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.means_

    # Draw white circles at cluster centers
    plt.scatter(centers[:, 10], centers[:, 8], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax.scatter( c[10], c[8], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax.set_title("The visualization of the clustered data.")
    ax.set_xlabel("Feature space for the 1st feature")
    ax.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Clusters plot for EM clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()

#################################################
#PCA feature transformation
    
pca = PCA(n_components=11, random_state=10)
X_r = pca.fit(X).transform(X)
X_pca = X_r
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ["b","g","r","c","m","y","k"]
lw = 2

for color, i in zip(colors, [4,8]):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=i)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Wine Quality dataset')

#################################################
#ICA feature transformation
    
ica = FastICA(n_components=11, random_state=10)
X_r = ica.fit(X).transform(X)
X_ica = X_r

plt.figure()
colors = ["b","g","r","c","m","y","k"]
lw = 2

for color, i in zip(colors, [4,8]):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=i)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('ICA of Wine Quality dataset')

#################################################
#Random Projection feature transformation

rca = GaussianRandomProjection(n_components=11, random_state=10)
X_r = rca.fit_transform(X)
X_rca = X_r

plt.figure()
colors = ["b","g","r","c","m","y","k"]
lw = 2

for color, i in zip(colors, [4,8]):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=i)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Random Projection of Wine Quality dataset')

#################################################
#Univariate feature selection (K best)

from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif

X_new = SelectKBest(chi2, k=5).fit_transform(X, y)
X_fs = X_new

plt.figure()
colors = ["b","g","r","c","m","y","k"]
lw = 2

for color, i in zip(colors, [4,8]):
    plt.scatter(X_new[y == i, 4], X_new[y == i, 0], color=color, alpha=.8, lw=lw, label=i)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Chi square feature selection of Wine Quality dataset')

#################################################
#Rerun clustering on transformed features
range_n_clusters = [2,4,6,8]
X_test=pd.DataFrame(X_pca)
for n_clusters in range_n_clusters:
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    ax = fig.add_subplot(111)
    
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X_test)
    cluster_labels = clusterer.labels_

    silhouette_avg = silhouette_score(X_test, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    print("The NMI score is: %.6f" % normalized_mutual_info_score(y, cluster_labels))
    
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax.scatter( X_test.iloc[:, 0], X_test.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    centers = clusterer.cluster_centers_

    ax.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax.scatter( c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax.set_title("The visualization of the clustered data.")
    ax.set_xlabel("Feature space for the 1st feature")
    ax.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("KMeans clustering using PCA feature transformation "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()
    
X_test=pd.DataFrame(X_fs)
for n_clusters in range_n_clusters:
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    ax = fig.add_subplot(111)

    clusterer = GaussianMixture(n_components=n_clusters, random_state=10).fit(X_test)
    cluster_labels = clusterer.predict(X_test)
    print("NMI score: %.6f" % normalized_mutual_info_score(y, cluster_labels))

    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    plt.scatter( X_test.iloc[:, 0], X_test.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    centers = clusterer.means_

    plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax.scatter( c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax.set_title("The visualization of the clustered data.")
    ax.set_xlabel("Feature space for the 1st feature")
    ax.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(("Clusters plot for EM clustering on PCA data "
                  "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')

    plt.show()

#################################################
#Rerun ANN on transformed features
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

clf = MLPClassifier(hidden_layer_sizes=(20, 5), random_state=0, solver="lbfgs")
plot_learning_curve(clf, "MLP using PCA transformed features", X_pca, y, ylim=[0,1])
plot_learning_curve(clf, "MLP using ICA transformed features", X_ica, y, ylim=[0,1])
plot_learning_curve(clf, "MLP using RCA transformed features", X_rca, y, ylim=[0,1])
plot_learning_curve(clf, "MLP using Selected 5 features", X_fs, y, ylim=[0,1])

#################################################
#Rerun ANN on transformed features with clusters new feature

clf = MLPClassifier(hidden_layer_sizes=(20, 5), random_state=0, solver="lbfgs")

clusterer = KMeans(n_clusters=6, random_state=10).fit(X_pca)
y_kmeans = clusterer.labels_
X_df = pd.DataFrame(X_pca)
X_df[11] = y_kmeans
plot_learning_curve(clf, "MLP using PCA transformed features", X_df, y, ylim=[0,1])

clusterer = KMeans(n_clusters=6, random_state=10).fit(X_ica)
y_kmeans = clusterer.labels_
X_df = pd.DataFrame(X_ica)
X_df[11] = y_kmeans
plot_learning_curve(clf, "MLP using ICA transformed features", X_df, y, ylim=[0,1])

clusterer = KMeans(n_clusters=6, random_state=10).fit(X_rca)
y_kmeans = clusterer.labels_
X_df = pd.DataFrame(X_rca)
X_df[11] = y_kmeans
plot_learning_curve(clf, "MLP using RCA transformed features", X_df, y, ylim=[0,1])

clusterer = KMeans(n_clusters=6, random_state=10).fit(X_fs)
y_kmeans = clusterer.labels_
X_df = pd.DataFrame(X_fs)
X_df[11] = y_kmeans
plot_learning_curve(clf, "MLP using selected 5 features", X_df, y, ylim=[0,1])

#################################################
#Data set 2: Gene expression data set
from sklearn.preprocessing import quantile_transform

data = pd.read_csv('sle_data.csv')
X = data.iloc[:, 1:5090]
y = np.append(np.repeat("HC",34), np.repeat("Disease",42))
features = list(X.columns.values)

scaler = MinMaxScaler(feature_range=[0,100])
scaler.fit(X)
X_norm = pd.DataFrame(quantile_transform(X))

#################################################
#Clustering, K means and EM
    
range_n_clusters = list(range(1,20))
sse = []
nmi = []
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    cluster_labels = clusterer.labels_
    sse.append(clusterer.inertia_)
    nmi.append(normalized_mutual_info_score(y, cluster_labels))
    
plt.plot(range_n_clusters, sse, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Errors')
plt.title('The Elbow Method showing the optimal k')
plt.show()

plt.plot(range_n_clusters, nmi, 'bx-')
plt.xlabel('k')
plt.ylabel('Normalized Mutual Information')
plt.title('The NMI metric showing the optimal k')
plt.show()

range_n_clusters = list(range(1,6))
nmi = []
for n_clusters in range_n_clusters:
    clusterer = GaussianMixture(n_components=n_clusters, random_state=0).fit(X)
    cluster_labels = clusterer.predict(X)
    nmi.append(normalized_mutual_info_score(y, cluster_labels))

plt.plot(range_n_clusters, nmi, 'bx-')
plt.xlabel('N components')
plt.ylabel('Normalized Mutual Information')
plt.title('The NMI metric showing EM clustering')
plt.show()

n_clusters=3
clusterer = GaussianMixture(n_components=n_clusters, random_state=10).fit(X)
cluster_labels = clusterer.predict(X)
print("NMI score: %.6f" % normalized_mutual_info_score(y, cluster_labels))

# 2nd Plot showing the actual clusters formed
colors = cm.spectral(y.astype(float) / n_clusters)
plt.scatter( X.iloc[:, 3], X.iloc[:, 7], marker='.', s=90, lw=0, alpha=0.7,
            c=colors, edgecolor='k')

# Labeling the clusters
centers = clusterer.means_

# Draw white circles at cluster centers
#plt.scatter(centers[:, 3], centers[:, 7], marker='o',
#            c="white", alpha=1, s=200, edgecolor='k')

for i, c in enumerate(centers):
    ax.scatter( c[3], c[7], marker='$%d$' % i, alpha=1,
                s=50, edgecolor='k')

ax.set_title("The visualization of the clustered data.")
ax.set_xlabel("Feature space for the 1st feature")
ax.set_ylabel("Feature space for the 2nd feature")

plt.suptitle(("EM clustering on raw sample data "
              "with n_clusters = %d" % n_clusters),
             fontsize=14, fontweight='bold')

plt.show()

#################################################
#PCA Feature transformation

pca = PCA(n_components=10, random_state=10)
X_r = pca.fit(X).transform(X)
X_pca = X_r
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ["b","g","r","c","m","y","k"]
lw = 2

for color, i in zip(colors, ["HC","Disease"]):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=i)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Disease/Health data set')

#################################################
#ICA Feature transformation

ica = FastICA(n_components=10, random_state=10)
X_r = ica.fit(X).transform(X)
X_ica = X_r

plt.figure()
colors = ["b","g","r","c","m","y","k"]
lw = 2

for color, i in zip(colors, ["HC","Disease"]):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=i)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('ICA of Disease/Health data set')

#################################################
#Random Projection feature transformation

rca = GaussianRandomProjection(n_components=10, random_state=10)
X_r = rca.fit_transform(X)
X_rca = X_r

plt.figure()
colors = ["b","g","r","c","m","y","k"]
lw = 2

for color, i in zip(colors, ["HC","Disease"]):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=i)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Random Projection of Disease/Health data set')

#################################################
#Univariate feature selection (K best)

X_new = SelectKBest(chi2, k=10).fit_transform(X, y)
X_fs = X_new

plt.figure()
colors = ["b","g","r","c","m","y","k"]
lw = 2

for color, i in zip(colors, ["HC","Disease"]):
    plt.scatter(X_new[y == i, 1], X_new[y == i, 0], color=color, alpha=.8, lw=lw, label=i)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('Chi square feature selection of Disease/Health data set')

#################################################
#Rerun clustering on transformed features

range_n_clusters = [2,3,4,5,6]
X_test=pd.DataFrame(X_fs)
for n_clusters in range_n_clusters:
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    ax = fig.add_subplot(111)
    
    clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X_test)
    cluster_labels = clusterer.labels_

    silhouette_avg = silhouette_score(X_test, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    print("The NMI score is: %.6f" % normalized_mutual_info_score(y, cluster_labels))
    
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax.scatter( X_test.iloc[:, 0], X_test.iloc[:, 1], marker='.', s=200, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    centers = clusterer.cluster_centers_

    ax.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax.scatter( c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax.set_title("The visualization of the clustered data.")
    ax.set_xlabel("Feature space for the 1st feature")
    ax.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("KMeans clustering using Selected 10 genes "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show()
    
X_test=pd.DataFrame(X_fs)
for n_clusters in range_n_clusters:
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    ax = fig.add_subplot(111)

    clusterer = GaussianMixture(n_components=n_clusters, random_state=10).fit(X_test)
    cluster_labels = clusterer.predict(X_test)
    print("NMI score: %.6f" % normalized_mutual_info_score(y, cluster_labels))

    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    plt.scatter( X_test.iloc[:, 0], X_test.iloc[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    centers = clusterer.means_

    plt.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax.scatter( c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax.set_title("The visualization of the clustered data.")
    ax.set_xlabel("Feature space for the 1st feature")
    ax.set_ylabel("Feature space for the 2nd feature")
    plt.suptitle(("Clusters plot for EM clustering on PCA data "
                  "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')

    plt.show()

#################################################
#Rerun ANN on transformed features

clf = MLPClassifier(hidden_layer_sizes=(20, 5), random_state=0, solver="lbfgs")
plot_learning_curve(clf, "MLP using FS transformed expression", X_fs, y, ylim=[0,1])

clf = MLPClassifier(hidden_layer_sizes=(20, 5), random_state=0, solver="lbfgs")
clusterer = KMeans(n_clusters=6, random_state=10).fit(X_pca)
y_kmeans = clusterer.labels_
X_df = pd.DataFrame(X_pca)
X_df[11] = y_kmeans
plot_learning_curve(clf, "MLP using PCA transformed features", X_df, y, ylim=[0,1])
