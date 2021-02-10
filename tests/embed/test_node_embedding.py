from bluegraph.backends.stellargraph import StellarGraphNodeEmbedder
from bluegraph.backends.neo4j import Neo4jNodeEmbedder, pgframe_to_neo4j, Neo4jGraphView


def _execute(driver, query):
    session = driver.session()
    result = session.run(query)
    result = result.data()
    session.close()
    return result


def _get_embedding_props(neo4j_driver, node_label, prop_name):
    query = (
        f"MATCH (n:{node_label}) "
        f"RETURN n.id as node_id, n.{prop_name} as emb"
    )
    result = _execute(neo4j_driver, query)
    return {el["node_id"]: el["emb"] for el in result}


def test_stellar_node_embedder(node_embedding_test_graph,
                               node_embedding_prediction_test_graph,):
    embedder = StellarGraphNodeEmbedder("complex")
    embedding = embedder.fit_model(
        pgframe=node_embedding_test_graph,
        embedding_dimension=6, epochs=5)
    assert(len(embedding["embedding"].iloc[0]) == 6)
    assert(
        set(node_embedding_test_graph.nodes()) ==
        set(embedding.index))

    embedder = StellarGraphNodeEmbedder("distmult")
    embedding = embedder.fit_model(
        pgframe=node_embedding_test_graph,
        embedding_dimension=10, epochs=5)
    assert(len(embedding["embedding"].iloc[0]) == 10)
    assert(
        set(node_embedding_test_graph.nodes()) ==
        set(embedding.index))

    embedder = StellarGraphNodeEmbedder(
        "attri2vec", feature_props=["age", "height", "weight"])
    node_embedding_test_graph.nodes(raw_frame=True)
    embedding = embedder.fit_model(
        pgframe=node_embedding_test_graph,
        embedding_dimension=3, epochs=5,
        length=5, number_of_walks=3)
    assert(len(embedding["embedding"].iloc[0]) == 3)
    assert(
        set(node_embedding_test_graph.nodes()) ==
        set(embedding.index))

    embeddings = embedder.predict_embeddings(
        node_embedding_prediction_test_graph
    )
    assert(len(embeddings) == 4)
    assert(
        set(node_embedding_prediction_test_graph.nodes()) ==
        set(embeddings.index))

    embedder = StellarGraphNodeEmbedder(
        "graphsage", feature_props=["age", "height", "weight"])
    node_embedding_test_graph.nodes(raw_frame=True)
    embedding = embedder.fit_model(
        pgframe=node_embedding_test_graph,
        embedding_dimension=3, epochs=5,
        length=5, number_of_walks=3)
    assert(len(embedding["embedding"].iloc[0]) == 3)
    assert(
        set(node_embedding_test_graph.nodes()) ==
        set(embedding.index))

    embeddings = embedder.predict_embeddings(
        node_embedding_prediction_test_graph
    )
    assert(len(embeddings) == 4)
    assert(
        set(node_embedding_prediction_test_graph.nodes()) ==
        set(embeddings.index))

    embedder.save("stellar_sage_emedder")
    embedder = StellarGraphNodeEmbedder.load(
        "stellar_sage_emedder.zip")
    embedder.info()


def test_neo4j_node_embedder(node_embedding_test_graph,
                             node_embedding_prediction_test_graph,
                             neo4j_driver):

    node_label = "TestPerson"
    edge_label = "TEST_KNOWS"

    # Populate neo4j with the test property graph
    pgframe_to_neo4j(
        pgframe=node_embedding_test_graph, driver=neo4j_driver,
        node_label=node_label, edge_label=edge_label)

    # Testing node2vec stream
    embedder = Neo4jNodeEmbedder("node2vec")
    embedding = embedder.fit_model(
        driver=neo4j_driver, node_label=node_label, edge_label=edge_label,
        embeddingDimension=6, walkLength=20, iterations=1)
    assert(len(embedding["embedding"].iloc[0]) == 6)

    # Alternatively, create a graph view
    graph_view = Neo4jGraphView(
        neo4j_driver, node_label=node_label, edge_label=edge_label)

    # Testing node2vec write
    embedder.fit_model(
        graph_view=graph_view, edge_weight="weight",
        write=True, write_property="node2vec",
        embeddingDimension=6, walkLength=20, iterations=1)
    emb = _get_embedding_props(neo4j_driver, node_label, "node2vec")
    assert(len(emb) == 7)
    assert(set(embedding.index) == set(emb.keys()))

    # Testing fastrp stream
    embedder = Neo4jNodeEmbedder("fastrp")
    embedding = embedder.fit_model(
        graph_view=graph_view,
        embeddingDimension=25)
    assert(len(embedding["embedding"].iloc[0]) == 25)

    # Testing fastrp write
    embedding = embedder.fit_model(
        graph_view=graph_view,
        edge_weight="weight",
        embeddingDimension=16,
        write=True, write_property="fastrp")
    emb = _get_embedding_props(neo4j_driver, node_label, "fastrp")
    assert(len(list(emb.values())[0]) == 16)

    # Testing GraphSage train and stream predict
    embedder = Neo4jNodeEmbedder(
        "graphsage", feature_props=["age", "height", "weight"])
    embedding = embedder.fit_model(
        graph_view=graph_view,
        embeddingDimension=3
    )
    assert(len(emb) == 7)
    assert(set(embedding.index) == set(emb.keys()))
    assert(len(embedding["embedding"].iloc[0]) == 3)

    # Testing GraphSage write predicts (passing all the credentials)
    embedder.predict_embeddings(
        pgframe=node_embedding_prediction_test_graph,
        driver=neo4j_driver,
        node_label="TestPredictPerson",
        edge_label="TEST_PREDICT_KNOWS",
        write=True, write_property="graphsage"
    )
    emb = _get_embedding_props(neo4j_driver, "TestPredictPerson", "fastrp")
    assert(len(emb) == 4)
    assert(
        set(node_embedding_prediction_test_graph.nodes()) ==
        set(emb.keys()))

    test_graph_view = Neo4jGraphView(
        neo4j_driver,
        node_label="TestPredictPerson",
        edge_label="TEST_PREDICT_KNOWS")

    # Testing GraphSage write predicts (passing graph view)
    embedder.predict_embeddings(
        graph_view=test_graph_view,
        write=True, write_property="graphsage"
    )
    emb = _get_embedding_props(neo4j_driver, "TestPredictPerson", "fastrp")
    assert(len(emb) == 4)
    assert(
        set(node_embedding_prediction_test_graph.nodes()) ==
        set(emb.keys()))

    embedder.save("neo4j_sage_emedder")
    embedder = Neo4jNodeEmbedder.load("neo4j_sage_emedder.zip")
    embedder.info()
