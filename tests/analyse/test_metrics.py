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

"""Test the metrics package."""
import math
import pytest
import networkx as nx

from bluegraph.backends.networkx import NXMetricProcessor
from bluegraph.backends.graph_tool import GTMetricProcessor
from bluegraph.backends.neo4j import Neo4jMetricProcessor


def _benchmark_processor(processor):
    processor.degree_centrality()
    d = processor.degree_centrality(weight="mi")
    processor.degree_centrality(
        weight="mi", write=True, write_property="degree")

    processor.pagerank_centrality()
    p = processor.pagerank_centrality(weight="mi")
    processor.pagerank_centrality(
        weight="mi", write=True, write_property="pagerank")

    processor.betweenness_centrality()
    b = processor.betweenness_centrality(distance="distance")
    processor.betweenness_centrality(
        distance="distance", write=True, write_property="betweenness")

    processor.closeness_centrality()
    c = processor.closeness_centrality(distance="distance")
    processor.closeness_centrality(
        distance="distance", write=True, write_property="closeness")

    res = processor.compute_all_node_metrics(
        degree_weights=["mi"],
        pagerank_weights=["mi"],
        betweenness_weights=["distance"],
        closeness_weights=["distance"])
    return (d, p, b, c)


def assert_approx_equal_metrics(d1, d2):
    assert(
        set([k for k, v in d1.items() if math.isnan(v)]) ==
        set([k for k, v in d2.items() if math.isnan(v)])
    )
    assert(
        {
            k: pytest.approx(v, rel=1e-4)
            for k, v in d1.items() if not math.isnan(v)
        } == {
            k: pytest.approx(v, rel=1e-4)
            for k, v in d2.items() if not math.isnan(v)
        }
    )


def test_nx_processor(random_pgframe):
    processor = NXMetricProcessor(random_pgframe, directed=False)
    d, p, b, c = _benchmark_processor(processor)
    props = []
    nx_object = processor.graph
    for n in nx_object.nodes():
        props = nx_object.nodes[n].keys()
        break
    assert(
        set(props) == set(["degree", "weight", "pagerank", "betweenness", "closeness"]))

    assert_approx_equal_metrics(
        nx.get_node_attributes(nx_object, "degree"), d)
    assert_approx_equal_metrics(
        nx.get_node_attributes(nx_object, "pagerank"), p)
    assert_approx_equal_metrics(
        nx.get_node_attributes(nx_object, "betweenness"), b)
    assert_approx_equal_metrics(
        nx.get_node_attributes(nx_object, "closeness"), c)


def test_gt_processor(random_pgframe):
    processor = GTMetricProcessor(random_pgframe, directed=True)
    d, p, b, c = _benchmark_processor(processor)
    gt_object = processor.graph
    assert(
        set(gt_object.vertex_properties.keys()) ==
        set(["@id", "weight", "degree", "pagerank", "betweenness", "closeness"]))

    dd = dict(zip(
        gt_object.vertex_properties["@id"],
        gt_object.vertex_properties["degree"]))
    assert_approx_equal_metrics(dd, d)
    pp = dict(zip(
        gt_object.vertex_properties["@id"],
        gt_object.vertex_properties["pagerank"]))
    assert_approx_equal_metrics(pp, p)
    bb = dict(zip(
        gt_object.vertex_properties["@id"],
        gt_object.vertex_properties["betweenness"]))
    assert_approx_equal_metrics(bb, b)
    cc = dict(zip(
        gt_object.vertex_properties["@id"],
        gt_object.vertex_properties["closeness"]))
    assert_approx_equal_metrics(cc, c)


def test_neo4j_processor(random_pgframe, neo4j_driver):
    processor = Neo4jMetricProcessor(
        pgframe=random_pgframe,
        driver=neo4j_driver,
        node_label="TestNode",
        edge_label="TestEdge",
        directed=False)
    d, p, b, c = _benchmark_processor(processor)

    processor._get_adjacency_matrix(
        random_pgframe.nodes(), weight="mi")
    processor._get_node_property_values(
        "weight", random_pgframe.nodes())

    # Assert all the node props are present
    query = "MATCH (n:TestNode) RETURN keys(n) AS props LIMIT 1"
    result = processor.execute(query)
    assert(
        set(result[0]["props"]) == set(
            ["id", "weight", "degree", "pagerank", "betweenness", "closeness"]))

    # Assert written props are equal to the streamed props
    query = "MATCH (n:TestNode) RETURN n.id AS node_id, n.degree as degree"
    result = processor.execute(query)
    assert_approx_equal_metrics(
        {record["node_id"]: record["degree"]
         for record in result}, d)

    query = "MATCH (n:TestNode) RETURN n.id AS node_id, n.pagerank as pagerank"
    result = processor.execute(query)
    assert_approx_equal_metrics(
        {record["node_id"]: record["pagerank"]
         for record in result}, p)

    query = "MATCH (n:TestNode) RETURN n.id AS node_id, n.betweenness as betweenness"
    result = processor.execute(query)
    assert_approx_equal_metrics(
        {
            record["node_id"]: record["betweenness"]
            for record in result
        }, b)

    query = "MATCH (n:TestNode) RETURN n.id AS node_id, n.closeness as closeness"
    result = processor.execute(query)
    assert_approx_equal_metrics(
        {
            record["node_id"]: record["closeness"]
            for record in result
        }, c)
