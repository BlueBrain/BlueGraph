# BlueGraph: unifying Python framework for graph analytics and co-occurrence analysis. 

# Copyright 2020-2021 Blue Brain Project / EPFL

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import warnings

from tensorflow.keras import optimizers, losses, metrics, Model
from tensorflow.keras.models import load_model

from stellargraph.data import UnsupervisedSampler
from stellargraph.layer import link_classification
from stellargraph.losses import graph_log_likelihood

from bluegraph.core.embed.embedders import (GraphElementEmbedder,
                                            DEFAULT_EMBEDDING_DIMENSION)

from .ml_utils import (_fit_node2vec,
                       _fit_deep_graph_infomax,
                       _dispatch_generator,
                       _dispatch_transductive_layer,
                       _dispatch_inductive_layer,
                       _generate_transductive_train_flow,
                       _dispatch_layer_sizes)
from ..io import pgframe_to_stellargraph


DEFAULT_STELLARGRAPH_PARAMS = {
    "embedding_dimension": DEFAULT_EMBEDDING_DIMENSION,
    "batch_size": 20,
    "negative_samples": 10,
    "epochs": 5,
    "length": 5,  # maximum length of a random walk
    "number_of_walks": 4,  # number of random walks per root node
    "num_samples": [10, 5],
    "random_walk_p": 0.5,  # Defines (unormalised) probability, 1/p, of returning to source node
    "random_walk_q": 2.0,  # Defines (unormalised) probability, 1/q, for moving away from source node
    "clusters": 2,
    "clusters_q": 1
}


STELLARGRAPH_PARAMS = {
    "transductive": [
        "embedding_dimension",
        "batch_size",
        "negative_samples",
        "epochs",
        "length",
        "num_samples",
        "number_of_walks",
        "random_walk_p",
        "random_walk_q"
    ],
    "inductive": [
        "embedding_dimension",
        "length",
        "number_of_walks",
        "batch_size",
        "epochs",
        "num_samples",
        "clusters",  # number of random clusters
        "clusters_q"  # number of clusters to combine for each mini-batch
    ]
}

LOSSES = {
    "complex": losses.BinaryCrossentropy(from_logits=True),
    "distmult": losses.BinaryCrossentropy(from_logits=True),
    "watchyourstep": graph_log_likelihood
}


class StellarGraphNodeEmbedder(GraphElementEmbedder):
    """Embedder for StellarGraph library."""

    _transductive_models = [
        "node2vec", "watchyourstep",
        "complex", "distmult",
        "gcn_dgi", "gat_dgi", "graphsage_dgi"
    ]
    _inductive_models = [
        "attri2vec", "graphsage",
        "cluster_gcn_dgi", "cluster_gat_dgi"
    ]

    @staticmethod
    def _generate_graph(pgframe, graph_configs):
        """Generate backend-specific graph object."""
        return pgframe_to_stellargraph(
            pgframe,
            directed=graph_configs["directed"],
            include_type=graph_configs["include_type"],
            feature_props=graph_configs["feature_props"],
            feature_vector_prop=graph_configs["feature_vector_prop"],
            edge_weight=graph_configs["edge_weight"]
        )

    def _dispatch_model_params(self, **kwargs):
        """Dispatch training parameters."""
        model_type = (
            "transductive"
            if self.model_name in StellarGraphNodeEmbedder._transductive_models
            else "inductive"
        )

        params = {}
        for k, v in kwargs.items():
            if k not in STELLARGRAPH_PARAMS[model_type]:
                warnings.warn(
                    f"StellarGraphNodeEmbedder's model '{self.model_name}' "
                    f"does not support the training parameter '{k}', "
                    "the parameter will be ignored",
                    GraphElementEmbedder.FittingWarning)
            else:
                params[k] = v

        for k, v in DEFAULT_STELLARGRAPH_PARAMS.items():
            if k not in params:
                params[k] = v

        return params

    def _fit_transductive_embedder(self, train_graph):
        """Fit transductive embedder (no model, just embeddings)."""
        if self.model_name == "node2vec":
            return _fit_node2vec(
                train_graph, self.params,
                edge_weight=self.graph_configs["edge_weight"])

        if self.model_name in ["gcn_dgi", "gat_dgi", "graphsage_dgi"]:
            return _fit_deep_graph_infomax(
                train_graph, self.params, self.model_name)

        generator = _dispatch_generator(
            train_graph, self.model_name, self.params)
        embedding_layer = _dispatch_transductive_layer(
            generator, self.model_name, self.params)

        x_inp, x_out = embedding_layer.in_out_tensors()

        # Create an embedding model
        model = Model(inputs=x_inp, outputs=x_out)
        model.compile(
            optimizer=optimizers.Adam(lr=0.001),
            loss=LOSSES[self.model_name],
            metrics=[metrics.BinaryAccuracy(threshold=0.0)],
        )

        # Train the embedding model
        train_generator = _generate_transductive_train_flow(
            train_graph, generator, self.model_name, self.params)

        model.fit(train_generator, epochs=self.params["epochs"], verbose=0)
        if self.model_name == "watchyourstep":
            embeddings = embedding_layer.embeddings()[0]
        else:
            embeddings = embedding_layer.embeddings()[0]
        return embeddings

    def _fit_inductive_embedder(self, train_graph):
        """Fit inductive embedder (predictive model and embeddings)."""
        if self.model_name in [
                "cluster_gcn_dgi", "cluster_gat_dgi"
           ]:
            return _fit_deep_graph_infomax(
                train_graph, self.params, self.model_name)

        unsupervised_samples = UnsupervisedSampler(
            train_graph,
            nodes=train_graph.nodes(),
            length=self.params["length"],
            number_of_walks=self.params["number_of_walks"]
        )

        generator = _dispatch_generator(
            train_graph, self.model_name, self.params,
            generator_type="edge")
        layer_sizes = _dispatch_layer_sizes(
            self.model_name, self.params)
        embedding_layer = _dispatch_inductive_layer(
            layer_sizes, generator, self.model_name, self.params)

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
            epochs=self.params["epochs"],
            shuffle=True, verbose=0)

        if self.model_name == "attri2vec":
            x_inp_src = x_inp[0]
        elif self.model_name == "graphsage":
            x_inp_src = x_inp[0::2]

        x_out_src = x_out[0]

        embedding_model = Model(inputs=x_inp_src, outputs=x_out_src)
        return embedding_model

    def _predict_embeddings(self, graph):
        node_generator = _dispatch_generator(
            graph, self.model_name, self.params).flow(graph.nodes())

        node_embeddings = self._embedding_model.predict(node_generator)
        res = dict(zip(graph.nodes(), node_embeddings.tolist()))
        return res

    @staticmethod
    def _save_predictive_model(model, path):
        model.save(path)

    @staticmethod
    def _load_predictive_model(path):
        return load_model(path, compile=False)
