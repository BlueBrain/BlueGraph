from bluegraph.backends.graph_tool import GTGraphProcessor
from bluegraph.backends.networkx import NXGraphProcessor
from bluegraph.backends.neo4j import Neo4jGraphProcessor


def _test_processor(processor, test_nodes, test_edges):
    nodes = processor.nodes()
    assert(len(test_nodes) == len(nodes))
    assert(len(processor.get_node(nodes[0])) > 0)
    new_node = str(nodes[0]) + "_la"
    processor.rename_nodes({nodes[0]: new_node})
    assert(new_node in processor.nodes())
    processor.set_node_properties(new_node, {"new_prop": "hello"})
    assert("new_prop" in processor.get_node(new_node))
    edges = processor.edges()
    assert(len(test_edges) == len(edges))
    edge = edges[0]
    assert(len(processor.get_edge(edge[0], edge[1])) > 0)
    # add_ede
    processor.set_edge_properties(
        edge[0], edge[1], {"new_edge_prop": "hello"})
    assert("new_edge_prop" in processor.get_edge(edge[0], edge[1]))

    edge = processor.edges()[0]
    processor.remove_edge(edge[0], edge[1])
    processor.add_edge(edge[0], edge[1])
    assert(edge in processor.edges())
    gr = processor.subgraph(nodes_to_include=processor.nodes()[:5])
    subprocessor = processor.__class__.from_graph_object(gr)
    if not isinstance(processor, Neo4jGraphProcessor):
        assert(len(subprocessor.nodes()) == 5)
    assert(len(processor.neighbors(new_node)) > 0)
    frame = processor.get_pgframe()
    assert(len(frame.nodes()) == len(processor.nodes()))
    assert(len(frame.edges()) == len(processor.edges()))
    processor.remove_node(new_node)


def test_gt_processor(random_pgframe):
    test_nodes = random_pgframe.nodes()
    test_edges = random_pgframe.edges()
    processor = GTGraphProcessor(random_pgframe)
    _test_processor(processor, test_nodes, test_edges)

    processor = GTGraphProcessor(random_pgframe, directed=True)
    _test_processor(processor, test_nodes, test_edges)


def test_nx_processor(random_pgframe):
    test_nodes = random_pgframe.nodes()
    test_edges = random_pgframe.edges()
    processor = NXGraphProcessor(random_pgframe)
    _test_processor(processor, test_nodes, test_edges)

    processor = NXGraphProcessor(random_pgframe, directed=True)
    _test_processor(processor, test_nodes, test_edges)


def test_neo4j_processor(random_pgframe, neo4j_driver):
    test_nodes = random_pgframe.nodes()
    test_edges = random_pgframe.edges()
    processor = Neo4jGraphProcessor(
        pgframe=random_pgframe,
        driver=neo4j_driver,
        node_label="TestProcessingNode",
        edge_label="TestProcessingEdge",
        directed=False)
    _test_processor(processor, test_nodes, test_edges)
    processor._get_identity_view()._clear()
    processor = Neo4jGraphProcessor(
        pgframe=random_pgframe,
        driver=neo4j_driver,
        node_label="TestProcessingNode",
        edge_label="TestProcessingEdge",
        directed=True)
    _test_processor(processor, test_nodes, test_edges)
