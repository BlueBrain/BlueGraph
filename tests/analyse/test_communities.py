import numpy as np
import networkx as nx

from bluegraph.core.analyse.communities import CommunityDetector
from bluegraph.backends.networkx import NXCommunityDetector
from bluegraph.backends.neo4j import Neo4jCommunityDetector


def _benchmark_comminities(detector, community_test_graph):
    louvain_partition = detector.detect_communities(strategy="louvain")
    detector.evaluate_parition(louvain_partition)
    detector.evaluate_parition(louvain_partition, metric="coverage")
    detector.evaluate_parition(louvain_partition, metric="performance")
    assert(
        len(louvain_partition) == community_test_graph.number_of_nodes())
    assert(len(set(louvain_partition.values())) == 4)

    # Write version
    detector.detect_communities(
        strategy="louvain", write=True, write_property="louvain_community")

    gn_2_partition = None
    gn_4_partition = None
    gn_2_partition = detector.detect_communities(strategy="girvan-newman")
    detector.evaluate_parition(gn_2_partition)
    detector.evaluate_parition(gn_2_partition, metric="coverage")
    detector.evaluate_parition(gn_2_partition, metric="performance")
    assert(
        len(gn_2_partition) == community_test_graph.number_of_nodes())

    # Write version
    detector.detect_communities(
        strategy="girvan-newman", write=True, write_property="gn_2_community")

    gn_4_partition = detector.detect_communities(
        strategy="girvan-newman", n_communities=4)
    detector.evaluate_parition(gn_4_partition)
    detector.evaluate_parition(gn_4_partition, metric="coverage")
    detector.evaluate_parition(gn_4_partition, metric="performance")
    assert(
        len(gn_4_partition) == community_test_graph.number_of_nodes())

    # Write version
    detector.detect_communities(
        strategy="girvan-newman", n_communities=4, write=True,
        write_property="gn_4_community")

    try:
        gn_inter_partition = detector.detect_communities(
            strategy="girvan-newman", n_communities=4,
            intermediate=True)
        assert(
            len(gn_inter_partition) == community_test_graph.number_of_nodes())
        assert(len(list(gn_inter_partition.values())[0]) > 0)
    except TypeError:
        pass

    # Write version
    detector.detect_communities(
        strategy="girvan-newman", n_communities=4,
        intermediate=True, write=True,
        write_property="gn_4_int_community")

    lpa_partition = None
    hierarchical_partition = None

    lpa_partition = detector.detect_communities(strategy="lpa")
    detector.evaluate_parition(lpa_partition)
    detector.evaluate_parition(lpa_partition, metric="coverage")
    detector.evaluate_parition(lpa_partition, metric="performance")
    assert(
        len(lpa_partition) == community_test_graph.number_of_nodes())
    assert(len(set(lpa_partition.values())) >= 1)

    # Write version
    detector.detect_communities(
        strategy="lpa", write=True, write_property="lpa_community")

    hierarchical_partition = detector.detect_communities(
        strategy="hierarchical",
        feature_vectors=np.random.rand(
            community_test_graph.number_of_nodes(), 3),
        n_communities=5)
    assert(
        len(hierarchical_partition) == community_test_graph.number_of_nodes())
    assert(len(set(hierarchical_partition.values())) == 5)

    # Write version
    detector.detect_communities(
        strategy="hierarchical",
        feature_vectors=np.random.rand(
            community_test_graph.number_of_nodes(), 3),
        n_communities=5, write=True, write_property="h_community")

    return (
        louvain_partition, gn_2_partition, gn_4_partition,
        lpa_partition, hierarchical_partition
    )


def test_nx_communities(community_test_graph):
    # TODO:
    # - test weighted version
    coms = NXCommunityDetector(community_test_graph, directed=False)
    l, g2, g4, lpa, h = _benchmark_comminities(coms, community_test_graph)
    nx_object = coms.graph
    for n in nx_object.nodes():
        props = nx_object.nodes[n].keys()
        break
    assert(
        set([
            "louvain_community",
            "gn_2_community",
            "gn_4_community",
            "gn_4_int_community",
            "lpa_community",
            "h_community"]).issubset(set(props)))
    assert(
        len(set(l.values())) ==
        len(set(nx.get_node_attributes(nx_object, "louvain_community").values()))
    )
    assert(
        len(set(g2.values())) ==
        len(set(nx.get_node_attributes(nx_object, "gn_2_community").values()))
    )
    assert(
        len(set(g4.values())) ==
        len(set(nx.get_node_attributes(nx_object, "gn_4_community").values()))
    )
    assert(nx.get_node_attributes(nx_object, "gn_4_int_community"))
    assert(nx.get_node_attributes(nx_object, "lpa_community"))
    assert(nx.get_node_attributes(nx_object, "h_community"))


def test_neo4j_communities(community_test_graph, neo4j_driver):
    coms = Neo4jCommunityDetector(
        pgframe=community_test_graph,
        driver=neo4j_driver,
        node_label="TestNode",
        edge_label="TestEdge", directed=False)
    l, g2, g4, lpa, h = _benchmark_comminities(coms, community_test_graph)
    # partition = coms.detect_communities(strategy="louvain")
    # print(partition)
    # Assert all the node props are present
    query = "MATCH (n:TestNode) RETURN keys(n) AS props LIMIT 1"
    result = coms.execute(query)
    # assert(
    #     set([
    #         "louvain_community",
    #         "gn_2_community",
    #         "gn_4_community",
    #         "gn_4_int_community",
    #         "lpa_community",
    #         "h_community"]).issubset(set(result[0]["props"])))

    # Assert written props are equal to the streamed props
    query = "MATCH (n:TestNode) RETURN n.id AS node_id, n.louvain_community as louvain_community"
    result = coms.execute(query)
    assert(
        len(set(l.values())) ==
        len(set(
            {record["node_id"]: record["louvain_community"] for record in result}.values()))
    )
