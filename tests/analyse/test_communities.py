from bluegraph.backends.networkx import NXCommunityDetector


def test_nx_communities(community_test_graph):
    # TODO:
    # - test weighted version
    # - test write property
    coms = NXCommunityDetector(community_test_graph, directed=False)
    partition = coms.detect_communities(strategy="louvain")
    print(coms.evaluate_parition(partition))
    print(coms.evaluate_parition(partition, metric="coverage"))
    print(coms.evaluate_parition(partition, metric="performance"))
    assert(
        len(partition) == community_test_graph.number_of_nodes())
    assert(len(set(partition.values())) == 4)
    print()

    partition = coms.detect_communities(strategy="girvan-newman")
    print(coms.evaluate_parition(partition))
    print(coms.evaluate_parition(partition, metric="coverage"))
    print(coms.evaluate_parition(partition, metric="performance"))
    assert(
        len(partition) == community_test_graph.number_of_nodes())
    assert(len(set(partition.values())) == 2)
    print()

    partition = coms.detect_communities(
        strategy="girvan-newman", n_communities=4)
    print(coms.evaluate_parition(partition))
    print(coms.evaluate_parition(partition, metric="coverage"))
    print(coms.evaluate_parition(partition, metric="performance"))
    assert(
        len(partition) == community_test_graph.number_of_nodes())
    assert(len(set(partition.values())) == 4)

    partition = coms.detect_communities(
        strategy="girvan-newman", n_communities=4,
        intermediate=True)
    assert(
        len(partition) == community_test_graph.number_of_nodes())
    assert(len(list(partition.values())[0]) > 0)
