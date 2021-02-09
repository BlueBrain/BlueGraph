from bluegraph.backends.stellargraph import StellarGraphNodeEmbedder
from bluegraph.backends.neo4j import Neo4jNodeEmbedder


IMPLEMENTED_EMBEDDERS = {
    "stellargraph": [
        "complex", "distmult", "attri2vec", "graphsage"
    ],
    "neo4j": [
        "node2vec", "fastrp", "graphsage"
    ]
}


def _execute(driver, query):
    session = driver.session()
    result = session.run(query)
    result = result.data()
    session.close()
    return result


def test_stellar_node_embedder():
    pass


def test_neo4j_node_embedder(node_embedding_test_graph,
                             node_embedding_prediction_test_graph,
                             neo4j_driver):

    # params = {

    # }
    # embedder = Neo4jNodeEmbedder("node2vec", params)
    # embeddings = embedder.fit_model(
    #     node_embedding_test_graph, write=True, write_property="node2vec")
    # if embedder._embedding_model is not None:
    #     embedder.predict_embeddings(new_pgframe)
    #     embedder.save()
    # embedder = Neo4jNodeEmbedder.load()
    # embedder.info()

    node_label = "TestPerson"
    edge_label = "TEST_KNOWS"

    print("Testing node2vec stream....")
    embedder = Neo4jNodeEmbedder("node2vec")
    # embedder.info()
    embedding = embedder.fit_model(
        pgframe=node_embedding_test_graph,
        driver=neo4j_driver, node_label=node_label,
        edge_label=edge_label,
        embeddingDimension=6, walkLength=20, iterations=1)
    assert(len(embedding["embedding"].iloc[0]) == 6)

    print("Testing node2vec write....")
    embedder.fit_model(
        edge_weight="weight", write=True, write_property="node2vec",
        driver=neo4j_driver, node_label=node_label, edge_label=edge_label,
        embeddingDimension=6, walkLength=20, iterations=1)
    query = (
        f"MATCH (n:{node_label}) "
        "RETURN n.id as node_id, n.node2vec as emb"
    )
    result = _execute(neo4j_driver, query)
    emb = {el["node_id"]: el["emb"] for el in result}
    assert(len(emb) == 7)
    assert(set(embedding.index) == set(emb.keys()))

    print("Testing fastrp stream....")
    embedder = Neo4jNodeEmbedder("fastrp")
    # embedder.info()
    embedding = embedder.fit_model(
        driver=neo4j_driver, node_label=node_label,
        edge_label=edge_label,
        embeddingDimension=25)
    assert(len(embedding["embedding"].iloc[0]) == 25)
    embedding = embedder.fit_model(
        edge_weight="weight", driver=neo4j_driver,
        node_label=node_label,
        edge_label=edge_label,
        embeddingDimension=16)
    assert(len(embedding["embedding"].iloc[0]) == 16)

    print("Testing graphsage stream....")
    embedder = Neo4jNodeEmbedder(
        "graphsage", feature_props=["age", "height", "weight"])
    embedding = embedder.fit_model(
        driver=neo4j_driver, node_label=node_label,
        edge_label=edge_label,
        embeddingDimension=3
    )
    assert(len(emb) == 7)
    assert(set(embedding.index) == set(emb.keys()))
    assert(len(embedding["embedding"].iloc[0]) == 3)

    embedder.predict_embeddings(
        pgframe=node_embedding_prediction_test_graph,
        driver=neo4j_driver, node_label="TestPredictPerson",
        edge_label="TEST_PREDICT_KNOWS",
        write=True, write_property="graphsage"
    )
    query = (
        "MATCH (n:TestPredictPerson) "
        "RETURN n.id as node_id, n.graphsage as emb"
    )
    result = _execute(neo4j_driver, query)
    emb = {el["node_id"]: el["emb"] for el in result}
    assert(len(emb) == 4)
    assert(
        set(node_embedding_prediction_test_graph.nodes()) ==
        set(emb.keys()))
