"""Collection of utils for benchmarking embedding models."""
import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier


def transform_to_2d(node_embeddings):
    """Transform embeddings to the 2D space using TSNE."""
    transform = TSNE  # PCA
    trans = transform(n_components=2)
    node_embeddings_2d = trans.fit_transform(node_embeddings)
    return node_embeddings_2d


def cluster_nodes(node_embeddings, k=4):
    """Cluser nodes in the embedding space."""
    kmeans = KMeans(n_clusters=k, random_state=0).fit(node_embeddings)
    return kmeans.labels_


def plot_2d(nodes, node_types, node_embeddings_2d, node_colors, label_map=None,
            show_types=False, colors="NA"):
    """Plot a 2D representation of nodes."""
    alpha = 1
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    num_colors = len(label_map)

    cm = plt.get_cmap('gist_rainbow')

    generated_colors = np.array([
        cm(1. * c / num_colors) for c in label_map.values()
    ])
    np.random.shuffle(generated_colors)

    # create a scatter per node color
    for l, c in label_map.items():
        indices = np.where(np.array(node_colors) == c)
        ax.scatter(
            node_embeddings_2d[indices, 0],
            node_embeddings_2d[indices, 1],
            c=[generated_colors[c]] * indices[0].shape[0],
            cmap="jet",
            s=50,
            alpha=alpha,
            label=l
        )
    ax.legend()

    ax.set_title("TSNE visualization of node embeddings (colors={})".format(
        colors))
    plt.show()


def plot_2d_with_clusters(nodes, node_types, node_embeddings_2d, clusters,
                          show_types=False):
    """Draw the embedding points, coloring them by node types."""
    label_map = {l: i for i, l in enumerate(np.unique(clusters))}
    node_colors = [label_map[c] for c in clusters]
    plot_2d(
        nodes, node_types, node_embeddings_2d, node_colors, label_map,
        show_types, colors="clusters")


def plot_2d_with_types(nodes, node_types, node_embeddings_2d, show_types=False):
    """Draw the embedding points, coloring them by node types."""
    label_map = {l: i for i, l in enumerate(set(node_types.values()))}
    node_colors = [
        label_map[node_types[n]] for n in nodes]
    plot_2d(
        nodes, node_types, node_embeddings_2d,
        node_colors, label_map, show_types, colors="types")


def cluster_homogenity(nodes, node_types, clusters):
    """Calculate cluster homogeneity."""
    cluster_homs = []
    for c in np.unique(clusters):
        indices = np.where(np.array(clusters) == c)
        types = [node_types[n] for n in np.array(nodes)[indices]]
        (unique, counts) = np.unique(types, return_counts=True)
        max_count = max(counts)
        cluster_homs.append(max_count / len(types))
    return cluster_homs


def clusters_vs_types(graph, embedding, n_clusters=None, plot=True):
    """Benchmark embedding using clusters vs types test."""
    embedding_array = np.array(embedding["embedding"].to_list())

    nodes = graph.nodes()
    node_types = graph.get_node_typing(as_dict=True)
    if n_clusters is None:
        n_clusters = len(graph.node_types())

    node_embeddings_2d = transform_to_2d(embedding_array)

    if plot is True:
        plot_2d_with_types(
            nodes, node_types, node_embeddings_2d)

    clusters = cluster_nodes(embedding_array, n_clusters)
    ch = np.array(cluster_homogenity(nodes, node_types, clusters))
    print("Homogeneity per cluster: ", ch)
    print("Average cluster/type homogeneity: ", ch.mean())

    if plot is True:
        plot_2d_with_clusters(
            nodes, node_types, node_embeddings_2d, clusters, show_types=False)

    return ch.mean()


def predict_node_types(graph, embedding, test_size=0.3, model_name="svm"):
    """Benchmark embedding building predictive model for node types."""
    node_types = graph.get_node_typing()
    binarized_types = MultiLabelBinarizer().fit_transform(node_types)
    embedding_array = np.array(embedding["embedding"].to_list())

    # Split train/test set
    X_train, X_test, y_train, y_test = train_test_split(
        embedding_array, binarized_types,
        test_size=test_size, random_state=42)

    if model_name == "svm":
        model = OneVsRestClassifier(
            LinearSVC(random_state=0)).fit(X_train, y_train)
    elif model_name == "knn":
        model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
    else:
        raise ValueError("Unknown classification model type '{}'".format(
            model_name))

    answers = model.predict(X_test)

    macro_f1 = f1_score(y_test, answers, average='macro')
    micro_f1 = f1_score(y_test, answers, average='micro')

    print("Node type prediction model: ")
    print("\t Macro F1-score: ", macro_f1)
    print("\t Micro F1-score: ", micro_f1)

    return (macro_f1, micro_f1)


def generate_edge_samples(nodes, edges):
    """Generate train test edges with true and fake edges."""
    sample_edges = []

    for s, t in edges:
        sample_edges.append([s, t, 1])
        # chose a target node randomly
        random_target = np.random.choice(nodes, size=1)[0]
        sample_edges.append([s, random_target, 0])
    return np.array(sample_edges)


def predict_links(graph, embedding, test_size=0.3, model_name="svm"):
    """Benchmark embedding building predictive model for links."""
    train_true_edges, test_true_edges = train_test_split(
        graph.edges(), test_size=0.3)

    train_edges = generate_edge_samples(graph.nodes(), train_true_edges)
    test_edges = generate_edge_samples(graph.nodes(), test_true_edges)

    train_edge_features = (
        np.array(embedding.loc[train_edges[:, 0]]["embedding"].to_list()) -
        np.array(embedding.loc[train_edges[:, 1]]["embedding"].to_list())
    ) ** 2

    test_edge_features = (
        np.array(embedding.loc[test_edges[:, 0]]["embedding"].to_list()) -
        np.array(embedding.loc[test_edges[:, 1]]["embedding"].to_list())
    ) ** 2

    model = LinearSVC(random_state=0)
    model.fit(train_edge_features, train_edges[:, 2])

    prediction = model.predict(test_edge_features)

    macro_f1 = f1_score(test_edges[:, 2], prediction, average='macro')
    micro_f1 = f1_score(test_edges[:, 2], prediction, average='micro')

    print("Link prediction model (fake vs true links): ")
    print("\t Macro F1-score: ", macro_f1)
    print("\t Micro F1-score: ", micro_f1)

    print(macro_f1, micro_f1)


    # def get_similar_nodes(self, node_id, number=10, node_subset=None):
    #     """Get N most similar entities."""
    #     embeddings = self.embeddings
    #     if node_subset is not None:
    #         # filter embeddings
    #         embeddings = self.embeddings.loc[node_subset]
    #     if embeddings.shape[0] < number:
    #         number = embeddings.shape[0]
    #     search_vec = self.embeddings.loc[node_id]["embedding"]
    #     matrix = np.matrix(embeddings["embedding"].to_list())
    #     closest_indices = cKDTree(matrix).query(search_vec, k=number)[1]
    #     return embeddings.index[closest_indices].to_list()

