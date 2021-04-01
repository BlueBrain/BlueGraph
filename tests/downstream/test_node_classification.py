from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd

from bluegraph.backends.stellargraph import StellarGraphNodeEmbedder
from bluegraph.downstream.node_classification import NodeClassifier


def test_node_classification(random_pgframe):
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
    node2vec_classifier = NodeClassifier(
        LinearSVC(), feature_vector_prop="node2vec")

    types = ["Apple", "Orange", "Carrot"]
    node_types = pd.DataFrame([
        (str(n), np.random.choice(types, p=[0.5, 0.4, 0.1]))
        for n in range(random_pgframe.number_of_nodes())
    ], columns=["@id", "entity_type"])
    random_pgframe.add_node_properties(node_types)

    node2vec_classifier.fit(
        random_pgframe, train_elements=random_pgframe.nodes(),
        label_prop="entity_type")
    node2vec_pred = node2vec_classifier.predict(
        random_pgframe, predict_elements=random_pgframe.nodes())
