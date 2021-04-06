from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd

from bluegraph.backends.stellargraph import StellarGraphNodeEmbedder
from bluegraph.downstream.node_classification import NodeClassifier

from sklearn import model_selection
from bluegraph.downstream import (get_confusion_matrix,
                                  get_classification_scores,
                                  transform_to_2d,
                                  cluster_nodes,
                                  plot_2d)


def test_node_classification(random_pgframe):
    random_pgframe.rename_nodes({
        n: str(n)
        for n in random_pgframe.nodes()
    })

    train_nodes, test_nodes = model_selection.train_test_split(
        random_pgframe.nodes(), train_size=0.8)

    node2vec_embedder = StellarGraphNodeEmbedder(
        "node2vec", edge_weight="mi",
        embedding_dimension=10, length=5, number_of_walks=10)
    node2vec_embedding = node2vec_embedder.fit_model(random_pgframe)
    random_pgframe.add_node_properties(
        node2vec_embedding.rename(columns={"embedding": "node2vec"}))
    node2vec_classifier = NodeClassifier(
        LinearSVC(), feature_vector_prop="node2vec")

    types = ["Apple", "Orange", "Carrot"]
    node_types = pd.DataFrame([
        (str(n), np.random.choice(types, p=[0.5, 0.4, 0.1]))
        for n in range(random_pgframe.number_of_nodes())
    ], columns=["@id", "entity_type"])
    random_pgframe.add_node_properties(node_types)

    cluster_nodes(
        random_pgframe.get_node_property_values("node2vec").to_list())
    node2vec_2d = transform_to_2d(
        random_pgframe.get_node_property_values("node2vec").to_list())
    plot_2d(
        random_pgframe, vectors=node2vec_2d, label_prop="entity_type",
        silent=True)

    node2vec_classifier.fit(
        random_pgframe, train_elements=train_nodes,
        label_prop="entity_type")
    node2vec_pred = node2vec_classifier.predict(
        random_pgframe, predict_elements=test_nodes)

    true_labels = random_pgframe._nodes.loc[test_nodes, "entity_type"]
    scores = get_classification_scores(true_labels, node2vec_pred, multiclass=True)
    matrix = get_confusion_matrix(true_labels, node2vec_pred)
