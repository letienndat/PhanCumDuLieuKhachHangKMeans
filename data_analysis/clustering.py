from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer


def find_optimal_k(data, range_val=(1, 15)):
    inertia = []
    for n in range(range_val[0], range_val[1]):
        algorithm = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                           tol=0.0001, random_state=111, algorithm='elkan')
        algorithm.fit(data)
        inertia.append(algorithm.inertia_)
    return inertia


def check_k_finded(data):
    model = KMeans(random_state=1)
    visualizer = KElbowVisualizer(model, k=(1, 15))
    visualizer.fit(data)

    return visualizer


def apply_kmeans(data, k):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300,
                    tol=0.0001, random_state=111, algorithm='elkan')
    kmeans.fit(data)
    y_kmeans = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    return y_kmeans, centers


def silhouette_scores(data, range_val=(2, 15)):
    print('\n[SILHOUETTE SCORE]')
    for n in range(range_val[0], range_val[1]):
        kmeans = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300,
                        tol=0.0001, random_state=111, algorithm='elkan')
        y_kmeans = kmeans.fit_predict(data)
        print(f'Silhouette Score (k = {n}): {silhouette_score(data, y_kmeans)}')


def export_clustered_data(df, output_file):
    df.to_csv(output_file, index=False)
