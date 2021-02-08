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
    session.close()
    return result


def test_stellar_node_embedder():
    for model_name in IMPLEMENTED_EMBEDDERS["stellargraph"]:
        embedder = StellarGraphNodeEmbedder(model_name)
        print("\n")
        embedder.info()


def test_neo4j_node_embedder(node_embedding_test_graph,
                             neo4j_driver,
                             neo4j_test_node_label,
                             neo4j_test_edge_label):

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

    embedder = Neo4jNodeEmbedder("node2vec")
    # embedder.info()
    embedding = embedder.fit_model(
        pgframe=node_embedding_test_graph,
        driver=neo4j_driver, node_label=node_label,
        edge_label=edge_label,
        embeddingDimension=25, walkLength=20, iterations=4)
    assert(len(embedding["embedding"].iloc[0]) == 25)
    embedder.fit_model(
        edge_weight="weight", write=True, write_property="node2vec",
        driver=neo4j_driver, node_label=node_label,
        edge_label=edge_label,
        embeddingDimension=16, walkLength=20, iterations=4)

    embedder = Neo4jNodeEmbedder("fastrp")
    # embedder.info()
    embedding = embedder.fit_model(
        pgframe=node_embedding_test_graph,
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

    embedder = Neo4jNodeEmbedder(
        "graphsage", feature_props=["age", "height", "weight"])
    # embedder.info()
    embedding = embedder.fit_model(
        driver=neo4j_driver, node_label=node_label,
        edge_label=edge_label,
        embeddingDimension=3
    )

    # TODO: test if the embeddings are written
