from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd


from sklearn import model_selection
from bluegraph.downstream import (EmbeddingPipeline,
                                  get_classification_scores)
from bluegraph.backends.stellargraph import StellarGraphNodeEmbedder
from bluegraph.downstream.link_prediction import (generate_negative_edges,
                                                  EdgePredictor)


def test_link_prediction(random_pgframe):
    random_pgframe.rename_nodes({
        n: str(n)
        for n in random_pgframe.nodes()
    })

    node2vec_embedder = StellarGraphNodeEmbedder(
        "node2vec", edge_weight="mi",
        embedding_dimension=10, length=5, number_of_walks=10)
    node2vec_embedding = node2vec_embedder.fit_model(random_pgframe)
    random_pgframe.add_node_properties(
        node2vec_embedding.rename(columns={"embedding": "node2vec"}))

    false_edges = generate_negative_edges(random_pgframe)
    true_train_edges, true_test_edges = model_selection.train_test_split(
        random_pgframe.edges(), train_size=0.8)
    false_train_edges, false_test_edges = model_selection.train_test_split(
        false_edges, train_size=0.8)
    model = EdgePredictor(
        LinearSVC(), feature_vector_prop="node2vec",
        operator="hadamard", directed=False)
    model.fit(
        random_pgframe, true_train_edges,
        negative_samples=false_train_edges)
    true_labels = np.hstack([
        np.ones(len(true_test_edges)),
        np.zeros(len(false_test_edges))])
    y_pred = model.predict(random_pgframe, true_test_edges + false_test_edges)
    get_classification_scores(true_labels, y_pred)
