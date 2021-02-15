"""Test the metrics package."""
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
    return (d, p, b, c)


def assert_approx_equal_metrics(d1, d2):
    assert(
        {
            k: pytest.approx(v, rel=1e-4)
            for k, v in d1.items()
        } == {
            k: pytest.approx(v, rel=1e-4)
            for k, v in d2.items()
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
        set(props) == set(["degree", "pagerank", "betweenness", "closeness"]))

    assert_approx_equal_metrics(
        nx.get_node_attributes(nx_object, "degree"), d)
    assert_approx_equal_metrics(
        nx.get_node_attributes(nx_object, "pagerank"), p)
    assert_approx_equal_metrics(
        nx.get_node_attributes(nx_object, "betweenness"), b)
    assert_approx_equal_metrics(
        nx.get_node_attributes(nx_object, "closeness"), c)


def test_gt_processor(random_pgframe):
    processor = GTMetricProcessor(random_pgframe, directed=False)
    d, p, b, c = _benchmark_processor(processor)
    gt_object = processor.graph
    assert(
        set(gt_object.vertex_properties.keys()) ==
        set(["@id", "degree", "pagerank", "betweenness", "closeness"]))

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

    # Assert all the node props are present
    query = "MATCH (n:TestNode) RETURN keys(n) AS props LIMIT 1"
    result = processor.execute(query)
    assert(
        set(result[0]["props"]) == set(
            ["id", "degree", "pagerank", "betweenness", "closeness"]))

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
