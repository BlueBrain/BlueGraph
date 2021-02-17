from .utils import ElementClassifier

import math
import numpy as np


def generate_negative_edges(pgframe, p=0.5, directed=True, edges_to_exclude=None):
    if edges_to_exclude is None:
        edges_to_exclude = []
    n_edges = int((pgframe.number_of_edges() - len(edges_to_exclude)) * p)
    sources = list(pgframe.nodes())
    targets = list(pgframe.nodes())

    negative_edges = set()

    # Upper bound on the number of iterations
    t = math.ceil(
        n_edges /
        ((1 - pgframe.density(directed=directed)) * pgframe.number_of_nodes())) * 2

    existing_edges = set(pgframe.edges())
    if directed is False:
        existing_edges.update({(u[1], u[0]) for u in existing_edges})

    for _ in range(t):
        np.random.shuffle(sources)
        np.random.shuffle(targets)

        for s, t in zip(sources, targets):
            if s != t and (s, t) not in negative_edges and\
               (s, t) not in existing_edges:
                negative_edges.add((s, t))
            if len(negative_edges) == n_edges:
                break
        if len(negative_edges) == n_edges:
            break
    return list(negative_edges)


def hadamard(u, v):
    return u * v


def l1(u, v):
    return np.abs(u - v)


def l2(u, v):
    return (u - v) ** 2


def avg(u, v):
    return (u + v) / 2.0


BINARY_OPERATORS = {
    "hadamard": hadamard,
    "l1": l1,
    "l2": l2,
    "average": avg
}


class EdgePredictor(ElementClassifier):
    """A minimal interface for a edge prediction model.

    This wrapper alows to build predictive models of PGFrame edges. Such
    model distinguish between true and false edges.
    """
    def __init__(self, model, feature_vector_prop=None, feature_props=None,
                 operator="hadamard", directed=True):
        if operator not in ["hadamard", "l1", "l2", "average"]:
            raise ValueError()

        super().__init__(model, feature_vector_prop, feature_props)
        self.operator = operator
        self.directed = directed

    def _generate_train_elements(self, pgframe, elements=None,
                                 negative_samples=None, negative_p=0.5):
        if elements is None:
            elements = pgframe.edges()

        # If negative samples are not provided, generate using `negative_p`
        if negative_samples is None:
            negative_samples = generate_negative_edges(
                pgframe.subgraph(edges=elements),
                p=negative_p, directed=self.directed)
        return elements, negative_samples

    def _generate_predict_elements(self, pgfame, elements=None):
        if elements is None:
            raise ValueError(
                "Edges to predicted are not provided")
        return elements

    def _generate_train_labels(self, pgframe, elements, label_prop=None):
        true_edges, false_edges = elements
        labels = np.hstack([
            np.ones(len(true_edges)), np.zeros(len(false_edges))
        ])
        return labels

    def _generate_data_table(self, pgframe, elements):
        if isinstance(elements, tuple):
            true_edges, false_edges = elements
            elements = true_edges + false_edges
        edge_indices_array = np.array(elements)
        source_features = self._get_node_features(
            pgframe, edge_indices_array[:, 0])
        target_features = self._get_node_features(
            pgframe, edge_indices_array[:, 1])
        return BINARY_OPERATORS[self.operator](
            source_features,
            target_features)
