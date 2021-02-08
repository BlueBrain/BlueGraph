"""Collection of StellarGraph machine learning utils."""
import pandas as pd

from tensorflow.keras import optimizers, losses, metrics, regularizers, Model

from stellargraph.data import UnsupervisedSampler
from stellargraph.mapper import (KGTripleGenerator,
                                 Attri2VecLinkGenerator,
                                 Attri2VecNodeGenerator,
                                 GraphSAGELinkGenerator,
                                 GraphSAGENodeGenerator)
from stellargraph.layer import (ComplEx,
                                DistMult,
                                Attri2Vec,
                                GraphSAGE,
                                link_classification)


def fit_embedder(graph, params, model_name):
    """Fit a KG embedding model."""



def fit_attri_embedder(graph, params, model_name):
    """Fit attributed graph embedding model."""
    unsupervised_samples = UnsupervisedSampler(
        graph,
        nodes=graph.nodes(),
        length=params["length"],
        number_of_walks=params["number_of_walks"]
    )

    # Create a generator of data for embedding and the embedding layer
    if model_name == "attri2vec":
        generator = Attri2VecLinkGenerator(
            graph, params["batch_size"])

        layer_sizes = [params["embedding_dimension"]]

        embedding_layer = Attri2Vec(
            layer_sizes=layer_sizes,
            generator=generator,
            bias=False, normalize=None
        )
    elif model_name == "GraphSAGE":
        generator = GraphSAGELinkGenerator(
            graph,
            params["batch_size"],
            params["num_samples"])

        layer_sizes = [
            params["embedding_dimension"],
            params["embedding_dimension"]
        ]

        embedding_layer = GraphSAGE(
            layer_sizes=layer_sizes,
            generator=generator,
            bias=True,
            dropout=0.0,
            normalize="l2"
        )

    x_inp, x_out = embedding_layer.in_out_tensors()

    prediction = link_classification(
        output_dim=1, output_act="sigmoid", edge_embedding_method="ip"
    )(x_out)

    model = Model(inputs=x_inp, outputs=prediction)

    model.compile(
        optimizer=optimizers.Adam(lr=1e-3),
        loss=losses.binary_crossentropy,
        metrics=[metrics.binary_accuracy],
    )

    train_generator = generator.flow(unsupervised_samples)

    model.fit(
        train_generator,
        epochs=params["epochs"],
        shuffle=True)

    if model_name == "attri2vec":
        x_inp_src = x_inp[0]
    elif model_name == "GraphSAGE":
        x_inp_src = x_inp[0::2]

    x_out_src = x_out[0]

    embedding_model = Model(inputs=x_inp_src, outputs=x_out_src)
    return embedding_model



