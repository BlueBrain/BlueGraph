import pandas as pd
import random
from nltk.corpus import words

from bluegraph.preprocess.encoders import ScikitLearnPGEncoder
from bluegraph.preprocess.utils import TfIdfEncoder
from bluegraph.backends.stellargraph import StellarGraphNodeEmbedder
from bluegraph.backends.gensim import GensimNodeEmbedder
from bluegraph.downstream.similarity import (FaissSimilarityIndex,
                                             ScikitLearnSimilarityIndex,
                                             SimilarityProcessor)
from bluegraph.downstream import EmbeddingPipeline
from bluegraph.core.embed.embedders import GraphElementEmbedder


def test_embedding_pipeline(random_pgframe):
    random_pgframe.rename_nodes({
        n: str(n)
        for n in random_pgframe.nodes()
    })
    desc = pd.DataFrame(
        [
            (n, ' '.join(random.sample(words.words(), 20)))
            for n in random_pgframe.nodes()
        ],
        columns=["@id", "desc"]
    )
    random_pgframe.add_node_properties(desc, prop_type="text")

    # Create an encoder
    encoder = ScikitLearnPGEncoder(
        node_properties=["desc"],
        edge_features=False,
        heterogeneous=False,
        encode_types=True,
        drop_types=True,
        text_encoding="tfidf",
        standardize_numeric=True)

    # create an embedder
    D = 128
    params = {
        "length": 5,
        "number_of_walks": 10,
        "epochs": 5,
        "embedding_dimension": D
    }
    attri2vec_embedder = StellarGraphNodeEmbedder(
        "attri2vec", feature_vector_prop="features",
        edge_weight="mi", **params)

    sim_processor = SimilarityProcessor(
        similarity_index=FaissSimilarityIndex(
            similarity="cosine", dimension=D))

    attri2vec_pipeline = EmbeddingPipeline(
        preprocessor=encoder,
        embedder=attri2vec_embedder,
        similarity_processor=sim_processor)

    attri2vec_pipeline.run_fitting(random_pgframe)
    attri2vec_pipeline.save(
        "attri2vec_test_model",
        compress=True)

    pipeline = EmbeddingPipeline.load(
        "attri2vec_test_model.zip",
        embedder_interface=GraphElementEmbedder,
        embedder_ext="zip")

    pipeline.retrieve_embeddings(["0", "1"])
    pipeline.get_neighbors(existing_points=["0", "1"], k=5)


def test_poincare_embedding_pipeline(random_pgframe):
    random_pgframe.rename_nodes({
        n: str(n)
        for n in random_pgframe.nodes()
    })

    # Create an encoder
    D = 32
    params = {
        "size": 6,
        "epochs": 100,
        "negative": 3
    }
    embedder = GensimNodeEmbedder(
        "poincare", directed=True, **params)
    embedding = embedder.fit_model(random_pgframe)
    sim_processor = SimilarityProcessor(
        ScikitLearnSimilarityIndex(
            64, similarity="poincare", leaf_size=20,
            initial_vectors=embedding["embedding"].tolist()),
        point_ids=embedding.index)
    poincare_pipeline = EmbeddingPipeline(
        embedder=embedder,
        similarity_processor=sim_processor)

    # poincare_pipeline.run_fitting(random_pgframe)
    poincare_pipeline.save(
        "poincare_test_model",
        compress=True)

    pipeline = EmbeddingPipeline.load(
        "poincare_test_model.zip",
        embedder_interface=GraphElementEmbedder,
        embedder_ext="zip")

    pipeline.retrieve_embeddings(["0", "1"])
    pipeline.get_neighbors(existing_points=["0", "1"], k=5)


def test_embedding_pipeline_with_prediction(random_words):
    pipeline = EmbeddingPipeline(
        preprocessor=TfIdfEncoder({
            "max_features": 400,
            "analyzer": "char",
            "ngram_range": (3, 3),
        }),
        embedder=None,
        similarity_processor=SimilarityProcessor(
            similarity_index=FaissSimilarityIndex(
               similarity="euclidean", dimension=400, n_segments=10)
        )
    )
    pipeline.run_fitting(random_words, point_ids=random_words)
    assert(pipeline.is_inductive() and not pipeline.is_transductive())

    res = pipeline.retrieve_embeddings(
        random_words[:2] + ["hello i am not there"])
    assert(res[2] is None)
    table = pipeline.generate_embedding_table()
    assert(table.shape[0] == len(random_words))

    vectors = pipeline.run_prediction(["hello world"])
    ind, dist = pipeline.get_neighbors(vectors=vectors, k=10)
    assert(ind[0].shape[0] > 0)

    vectors = pipeline.run_prediction(
        ["hello world"], data_point_ids=["hello world"],
        add_to_index=True)
    assert("hello world" in pipeline.get_point_ids())

    new_ind, dist = pipeline.get_neighbors(
        existing_points=["hello world"])