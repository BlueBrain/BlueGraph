import pandas as pd
import warnings

from tensorflow.keras import optimizers, losses, metrics, regularizers, Model
from tensorflow.keras.models import load_model

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

from bluegraph.core.embed.embedders import (ElementEmbedder,
                                            DEFAULT_EMBEDDING_DIMENSION)

from ..io import pgframe_to_stellargraph


DEFAULT_STELLARGRAPH_PARAMS = {
    "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
    "batch_size": 20,
    "negative_samples": 10,
    "epochs": 5,
    "length": 5,
    "number_of_walks": 4,
    "num_samples": [10, 5],
}


STELLARGRAPH_PARAMS = {
    "transductive": [
        "embedding_dimension",
        "batch_size",
        "negative_samples",
        "epochs"
    ],
    "inductive": [
        "embedding_dimension",
        "length",
        "number_of_walks",
        "batch_size",
        "epochs",
        "num_samples"
    ]
}


def _dispatch_node_generator(graph, model_name, batch_size, num_samples):
    """Dispatch generator of nodes for the provided model name."""
    if model_name == "attri2vec":
        node_generator = Attri2VecNodeGenerator(
            graph, batch_size).flow(graph.nodes())
    elif model_name == "graphsage":
        node_generator = GraphSAGENodeGenerator(
            graph, batch_size, num_samples).flow(graph.nodes())
    else:
        raise ValueError("")

    return node_generator


class StellarGraphNodeEmbedder(ElementEmbedder):
    """Embedder for StellarGraph library."""

    _transductive_models = ["complex", "distmult"]
    _inductive_models = ["attri2vec", "graphsage"]

    @staticmethod
    def _generate_graph(pgframe, graph_configs):
        """Generate backend-specific graph object."""
        return pgframe_to_stellargraph(
            pgframe,
            directed=graph_configs["directed"],
            include_type=graph_configs["include_type"],
            feature_props=graph_configs["feature_props"],
            feature_vector_prop=graph_configs["feature_vector_prop"],
        )

    @staticmethod
    def _dispatch_training_params(model_name, defaults, **kwargs):
        """Dispatch training parameters."""
        model_type = (
            "transductive"
            if model_name in StellarGraphNodeEmbedder._transductive_models
            else "inductive"
        )

        params = {}
        for k, v in kwargs.items():
            if k not in STELLARGRAPH_PARAMS[model_type]:
                warnings.warn(
                    f"StellarGraphNodeEmbedder's model '{model_name}' "
                    f"does not support the training parameter '{k}', "
                    "the parameter will be ignored",
                    ElementEmbedder.FittingWarning)
            else:
                params[k] = v

        for k, v in DEFAULT_STELLARGRAPH_PARAMS.items():
            if k not in params:
                params[k] = v

        return params

    @staticmethod
    def _fit_transductive_embedder(train_graph, params, model_name):
        """Fit transductive embedder (no model, just embeddings)."""
        generator = KGTripleGenerator(train_graph, params["batch_size"])

        # Create an embedding layer
        if model_name == "distmult":
            embedding_layer = DistMult(
                generator,
                embedding_dimension=params["embedding_dimension"],
                embeddings_regularizer=regularizers.l2(1e-7),
            )
        elif model_name == "complex":
            embedding_layer = ComplEx(
                generator,
                embedding_dimension=params["embedding_dimension"],
                embeddings_regularizer=regularizers.l2(1e-7),
            )

        x_inp, x_out = embedding_layer.in_out_tensors()

        # Create an embedding model
        model = Model(inputs=x_inp, outputs=x_out)
        model.compile(
            optimizer=optimizers.Adam(lr=0.001),
            loss=losses.BinaryCrossentropy(from_logits=True),
            metrics=[metrics.BinaryAccuracy(threshold=0.0)],
        )

        # Train the embedding model
        train_generator = generator.flow(
            pd.DataFrame(
                train_graph.edges(
                    include_edge_type=True),
                columns=["source", "target", "label"]),
            negative_samples=params["negative_samples"],
            shuffle=True
        )

        model.fit(
            train_generator, epochs=params["epochs"])

        embeddings = embedding_layer.embeddings()[0]
        return embeddings

    @staticmethod
    def _fit_inductive_embedder(train_graph, params, model_name):
        """Fit inductive embedder (predictive model and embeddings)."""
        unsupervised_samples = UnsupervisedSampler(
            train_graph,
            nodes=train_graph.nodes(),
            length=params["length"],
            number_of_walks=params["number_of_walks"]
        )

        # Create a generator of data for embedding and the embedding layer
        if model_name == "attri2vec":
            generator = Attri2VecLinkGenerator(
                train_graph, params["batch_size"])

            layer_sizes = [params["embedding_dimension"]]

            embedding_layer = Attri2Vec(
                layer_sizes=layer_sizes,
                generator=generator,
                bias=False, normalize=None
            )
        elif model_name == "graphsage":
            generator = GraphSAGELinkGenerator(
                train_graph,
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
        elif model_name == "graphsage":
            x_inp_src = x_inp[0::2]

        x_out_src = x_out[0]

        embedding_model = Model(inputs=x_inp_src, outputs=x_out_src)
        return embedding_model

    def _predict_embeddings(self, graph, batch_size=None, num_samples=None,
                            **kwargs):
        if batch_size is None:
            batch_size = DEFAULT_STELLARGRAPH_PARAMS["batch_size"]
        if num_samples is None:
            num_samples = DEFAULT_STELLARGRAPH_PARAMS["num_samples"]

        node_generator = _dispatch_node_generator(
            graph, self.model_name, batch_size, num_samples)
        node_embeddings = self._embedding_model.predict(node_generator)
        res = dict(zip(graph.nodes(), node_embeddings.tolist()))
        return res

    @staticmethod
    def _save_predictive_model(model, path):
        model.save(path)

    @staticmethod
    def _load_predictive_model(path):
        return load_model(path, compile=False)
