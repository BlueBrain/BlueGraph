# BlueGraph: unifying Python framework for graph analytics and co-occurrence analysis. 

# Copyright 2020-2021 Blue Brain Project / EPFL

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import numpy as np
import networkx as nx

from bluegraph.core.analyse.communities import CommunityDetector
from bluegraph.backends.networkx import NXCommunityDetector
from bluegraph.backends.neo4j import Neo4jCommunityDetector
from bluegraph.backends.graph_tool import GTCommunityDetector


def _benchmark_comminities(detector, community_test_graph):
    louvain_partition = detector.detect_communities(strategy="louvain")
    detector.evaluate_parition(louvain_partition)
    detector.evaluate_parition(louvain_partition, metric="coverage")
    detector.evaluate_parition(louvain_partition, metric="performance")
    assert(
        len(louvain_partition) == community_test_graph.number_of_nodes())
    assert(len(set(louvain_partition.values())) == 4)
    assert(
        set([str(n) for n in louvain_partition.keys()]) ==
        set([str(n) for n in community_test_graph.nodes()]))

    # Weighted version
    weighted_louvain_partition = detector.detect_communities(
        strategy="louvain", weight="strength")
    assert(
        len(weighted_louvain_partition) == community_test_graph.number_of_nodes())

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

    # Weighted version
    weighted_gn_2_partition = detector.detect_communities(
        strategy="girvan-newman", weight="strength")
    assert(
        len(weighted_gn_2_partition) == community_test_graph.number_of_nodes())

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

        # Weighted version
        weighted_gn_inter_partition = detector.detect_communities(
            strategy="girvan-newman", weight="strength", n_communities=4,
            intermediate=True)
        assert(
            len(weighted_gn_inter_partition) ==
            community_test_graph.number_of_nodes())

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

    # Weighted version
    weighted_lpa_partition = detector.detect_communities(
        strategy="lpa", weight="strength")
    assert(
        len(weighted_lpa_partition) ==
        community_test_graph.number_of_nodes())

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

    # Weighted version
    weighted_hierarchical_partition = detector.detect_communities(
        strategy="hierarchical",
        feature_vectors=np.random.rand(
            community_test_graph.number_of_nodes(), 3),
        n_communities=5, weight="strength")
    assert(
        len(weighted_hierarchical_partition) ==
        community_test_graph.number_of_nodes())

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
        node_label="TestCommunityNode",
        edge_label="TestCommunityEdge", directed=False)
    l, g2, g4, lpa, h = _benchmark_comminities(coms, community_test_graph)

    # Assert all the node props are present
    query = "MATCH (n:TestCommunityNode) RETURN keys(n) AS props LIMIT 1"
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
    query = "MATCH (n:TestCommunityNode) RETURN n.id AS node_id, n.louvain_community as louvain_community"
    result = coms.execute(query)
    assert(
        len(set(l.values())) ==
        len(set(
            {
                record["node_id"]: record["louvain_community"]
                for record in result
            }.values()))
    )


def test_gt_communities(community_test_graph):
    coms = GTCommunityDetector(community_test_graph, directed=False)
    partition = coms.detect_communities(strategy="sbm")
    assert(
        len(partition) == community_test_graph.number_of_nodes())
    assert(len(set(partition.values())) == 1)
    partition = coms.detect_communities(strategy="sbm", min_communities=4)
    assert(
        len(partition) == community_test_graph.number_of_nodes())
    assert(len(set(partition.values())) == 4)

    partition = coms.detect_communities(strategy="sbm", nested=True)
    assert(
        len(partition) == community_test_graph.number_of_nodes())
    partition = coms.detect_communities(
        strategy="sbm", nested=True, min_communities=4)
    assert(
        len(partition) == community_test_graph.number_of_nodes())
