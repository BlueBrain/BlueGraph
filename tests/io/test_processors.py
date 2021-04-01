from bluegraph.backends.graph_tool import GTGraphProcessor


def _test_processor(processor, test_nodes, test_edges):
    nodes = processor.nodes()
    assert(len(test_nodes) == len(nodes))
    assert(len(processor.get_node(nodes[0])) > 0)
    processor.remove_node(nodes[0])

    edges = processor.edges()
    assert(len(test_edges) == len(edges))
    assert(len(processor.get_edge(edges[0][0], edges[0][1])) > 0)
    processor.remove_edge(edges[0][0], edges[0][1])


def test_gt_processor(random_pgframe):
    test_nodes = random_pgframe.nodes()
    test_edges = random_pgframe.edges()
    processor = GTGraphProcessor(random_pgframe)
    _test_processor(processor, test_nodes, test_edges)

    processor = GTGraphProcessor(random_pgframe, directed=True)
    _test_processor(processor, test_nodes, test_edges)


def test_nx_processor(random_pgframe):
    pass


def test_neo4j_processor(random_pgframe):
    pass

