import pytest

from bluegraph.backends.utils import (create_analyzer,
                                      create_node_embedder)
from bluegraph.backends.configs import ANALYZER_CLS


@pytest.mark.parametrize("analyzer,backend", [
    (a, b) for a in ANALYZER_CLS.keys() for b in ["networkx", "graph_tool"]])
def test_create_inmemory_analyzer(random_pgframe, analyzer, backend):
    analyzer = create_analyzer(analyzer, backend, random_pgframe)


@pytest.mark.parametrize("analyzer", list(ANALYZER_CLS.keys()))
def test_create_neo4j_analyzer(random_pgframe, analyzer, neo4j_driver):
    analyzer = create_analyzer(
        analyzer, "neo4j", random_pgframe, driver=neo4j_driver,
        node_label="TestFactoryNode", edge_label="TestFactoryEdge")


def test_create_node_embedder(node_embedding_test_graph, neo4j_driver):
    embedder = create_node_embedder(
        "stellargraph", "node2vec", embedding_dimension=6, length=10,
        number_of_walks=20, edge_weight="weight")
    embedder.fit_model(node_embedding_test_graph)

    embedder = create_node_embedder(
        "neo4j", "node2vec", embeddingDimension=6, walkLength=20,
        iterations=1, edge_weight="weight")
    embedder.fit_model(
        driver=neo4j_driver,
        node_label="TestFactoryNode", edge_label="TestFactoryEdge")
