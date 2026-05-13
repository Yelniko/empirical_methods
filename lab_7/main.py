from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd


def optim_number_claster(wine):
    lis = []
    r = range(1,11)

    for i in r:
        model = KMeans(n_clusters=i, random_state=42)
        model.fit(wine)
        lis.append(model.inertia_)

    plt.plot(r, lis, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

def main():
    wine_1 = load_wine()
    wine = wine_1.data
    wine = StandardScaler().fit_transform(wine)
    optim_number_claster(wine)

    kmeans = KMeans(n_clusters=3, random_state=42)
    y_pred = kmeans.fit_predict(wine)
    y_true = wine_1.target

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(wine)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    centers = pca.transform(kmeans.cluster_centers_)

    ax[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='viridis')
    ax[0].set_title('Real')
    ax[0].set_xlabel('PCA1')
    ax[0].set_ylabel('PCA2')

    ax[1].scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, cmap='viridis')
    ax[1].scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
    ax[1].set_title('KMeans')
    ax[1].set_xlabel('PCA1')
    ax[1].set_ylabel('PCA2')

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()