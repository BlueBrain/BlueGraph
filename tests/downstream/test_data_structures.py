import pandas as pd
import random
from nltk.corpus import words

from bluegraph.core import PandasPGFrame

from bluegraph.preprocess.encoders import ScikitLearnPGEncoder
from bluegraph.backends.stellargraph import StellarGraphNodeEmbedder
from bluegraph.downstream.similarity import SimilarityProcessor
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

    attri2vec_pipeline = EmbeddingPipeline(
        preprocessor=encoder,
        embedder=attri2vec_embedder,
        similarity_processor=SimilarityProcessor(
            similarity="cosine", dimension=D))

    attri2vec_pipeline.run_fitting(random_pgframe)
    attri2vec_pipeline.save(
        "attri2vec_test_model",
        compress=True)

    pipeline = EmbeddingPipeline.load(
        "attri2vec_test_model.zip",
        embedder_interface=GraphElementEmbedder,
        embedder_ext="zip")

    pipeline.retrieve_embeddings(["0", "1"])
    pipeline.get_similar_points(["0", "1"], k=5)
