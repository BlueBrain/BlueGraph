import numpy as np
from bluegraph.backends.networkx import NXCommunityDetector


def test_nx_communities(community_test_graph):
    # TODO:
    # - test weighted version
    # - test write property
    coms = NXCommunityDetector(community_test_graph, directed=False)
    partition = coms.detect_communities(strategy="louvain")
    coms.evaluate_parition(partition)
    coms.evaluate_parition(partition, metric="coverage")
    coms.evaluate_parition(partition, metric="performance")
    assert(
        len(partition) == community_test_graph.number_of_nodes())
    assert(len(set(partition.values())) == 4)

    partition = coms.detect_communities(strategy="girvan-newman")
    coms.evaluate_parition(partition)
    coms.evaluate_parition(partition, metric="coverage")
    coms.evaluate_parition(partition, metric="performance")
    assert(
        len(partition) == community_test_graph.number_of_nodes())
    assert(len(set(partition.values())) == 2)

    partition = coms.detect_communities(
        strategy="girvan-newman", n_communities=4)
    coms.evaluate_parition(partition)
    coms.evaluate_parition(partition, metric="coverage")
    coms.evaluate_parition(partition, metric="performance")
    assert(
        len(partition) == community_test_graph.number_of_nodes())
    assert(len(set(partition.values())) == 4)

    partition = coms.detect_communities(
        strategy="girvan-newman", n_communities=4,
        intermediate=True)
    assert(
        len(partition) == community_test_graph.number_of_nodes())
    assert(len(list(partition.values())[0]) > 0)

    partition = coms.detect_communities(strategy="lpa")
    coms.evaluate_parition(partition)
    coms.evaluate_parition(partition, metric="coverage")
    coms.evaluate_parition(partition, metric="performance")
    assert(
        len(partition) == community_test_graph.number_of_nodes())
    assert(len(set(partition.values())) > 1)

    partition = coms.detect_communities(
        strategy="hierarchical",
        feature_vectors=np.random.rand(
            community_test_graph.number_of_nodes(), 3),
        n_communities=5)
