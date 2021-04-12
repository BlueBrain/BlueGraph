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
"""Collection of StellarGraph machine learning utils.

StellarGraph demos have been used to implement the set of embedding utils
included in this module.

See the demos:
https://stellargraph.readthedocs.io/en/stable/demos/index.html#table-of-contents

"""
import pandas as pd

from tensorflow.keras import regularizers

from stellargraph.mapper import (KGTripleGenerator,
                                 Attri2VecLinkGenerator,
                                 Attri2VecNodeGenerator,
                                 GraphSAGELinkGenerator,
                                 GraphSAGENodeGenerator,
                                 AdjacencyPowerGenerator,
                                 CorruptedGenerator,
                                 FullBatchNodeGenerator,
                                 ClusterNodeGenerator)
from stellargraph.layer import (ComplEx,
                                DistMult,
                                Attri2Vec,
                                GraphSAGE,
                                WatchYourStep,
                                GCN, GAT, DeepGraphInfomax)
from stellargraph.data import BiasedRandomWalk
from gensim.models import Word2Vec

import tensorflow as tf
from tensorflow.keras import optimizers, Model


def _dispatch_generator(graph, model_name, params,
                        generator_type="node"):
    """Create a graph generator."""
    if model_name == "watchyourstep":
        return AdjacencyPowerGenerator(
            graph, num_powers=params["num_powers"])
    elif model_name in ["complex", "distmult"]:
        return KGTripleGenerator(graph, params["batch_size"])
    elif model_name == "attri2vec":
        if generator_type == "node":
            return Attri2VecNodeGenerator(
                graph, params["batch_size"])
        else:
            return Attri2VecLinkGenerator(
                graph, params["batch_size"])
    elif model_name in ["graphsage", "graphsage_dgi"]:
        if generator_type == "node":
            return GraphSAGENodeGenerator(
                graph, params["batch_size"], params["num_samples"])
        else:
            return GraphSAGELinkGenerator(
                graph, params["batch_size"], params["num_samples"])
    elif model_name in ["gcn_dgi", "gat_dgi"]:
        return FullBatchNodeGenerator(graph, sparse=False)
    elif model_name in ["cluster_gcn_dgi", "cluster_gat_dgi"]:
        return ClusterNodeGenerator(
            graph, clusters=params["clusters"],
            q=params["clusters_q"])
    else:
        raise ValueError(f"Unknown model name '{model_name}'")


def _generate_transductive_train_flow(train_graph, generator,
                                      model_name, params):
    if model_name == "watchyourstep":
        train_gen = generator.flow(
            batch_size=params["batch_size"], num_parallel_calls=10)
    elif model_name in ["complex", "distmult"]:
        train_gen = generator.flow(
            pd.DataFrame(
                train_graph.edges(
                    include_edge_type=True),
                columns=["source", "target", "label"]),
            negative_samples=params["negative_samples"],
            shuffle=True
        )
    return train_gen


def _dispatch_transductive_layer(generator, model_name, params):
    """Create an embedding layer."""
    embedding_layer = None
    if model_name == "watchyourstep":
        embedding_layer = WatchYourStep(
            generator,
            num_walks=params["number_of_walks"],
            embedding_dimension=params["embedding_dimension"],
            attention_regularizer=regularizers.l2(0.5),
        )
    elif model_name == "distmult":
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
    return embedding_layer


def _dispatch_layer_sizes(model_name, params):
    if model_name == "attri2vec":
        return [params["embedding_dimension"]]
    elif model_name == "graphsage":
        return [params["embedding_dimension"], params["embedding_dimension"]]


def _dispatch_inductive_layer(layer_sizes, generator, model_name, params):
    if model_name == "attri2vec":
        embedding_layer = Attri2Vec(
            layer_sizes=layer_sizes,
            generator=generator,
            bias=False, normalize=None
        )
    elif model_name == "graphsage":
        embedding_layer = GraphSAGE(
            layer_sizes=layer_sizes,
            generator=generator,
            bias=True,
            dropout=0.0,
            normalize="l2"
        )
    return embedding_layer


def _fit_node2vec(train_graph, params, edge_weight=None):
    rw = BiasedRandomWalk(train_graph)
    walks = rw.run(
        nodes=list(train_graph.nodes()),  # root nodes
        length=params["length"],
        n=params["number_of_walks"],
        p=params["random_walk_p"],
        q=params["random_walk_q"],
        weighted=edge_weight is not None
    )
    model = Word2Vec(walks, size=params["embedding_dimension"])
    return model.wv[train_graph.nodes()]


def _execute_deep_graph_infomax(train_graph, embedding_layer, generator, params):
    corrupted_generator = CorruptedGenerator(generator)
    gen = corrupted_generator.flow(train_graph.nodes())
    infomax = DeepGraphInfomax(embedding_layer, corrupted_generator)

    x_in, x_out = infomax.in_out_tensors()

    model = Model(inputs=x_in, outputs=x_out)
    model.compile(
        loss=tf.nn.sigmoid_cross_entropy_with_logits,
        optimizer=optimizers.Adam(lr=1e-3))
    model.fit(gen, epochs=params["epochs"], verbose=0)

    x_emb_in, x_emb_out = embedding_layer.in_out_tensors()

    # for full batch models, squeeze out the batch dim (which is 1)
    if generator.num_batch_dims() == 2:
        x_emb_out = tf.squeeze(x_emb_out, axis=0)

    embedding_model = Model(inputs=x_emb_in, outputs=x_emb_out)
    return embedding_model


def _fit_deep_graph_infomax(train_graph, params, model_name):
    """Train unsupervised Deep Graph Infomax."""
    if "gcn_dgi" in model_name or "gat_dgi" in model_name:
        if "cluster" in model_name:
            generator = ClusterNodeGenerator(
                train_graph, clusters=params["clusters"],
                q=params["clusters_q"])
        else:
            generator = FullBatchNodeGenerator(train_graph, sparse=False)

        if "gcn_dgi" in model_name:
            embedding_layer = GCN(
                layer_sizes=[params["embedding_dimension"]],
                activations=["relu"], generator=generator)
        elif "gat_dgi" in model_name:
            embedding_layer = GAT(
                layer_sizes=[params["embedding_dimension"]],
                activations=["relu"], generator=generator, attn_heads=8)
    elif model_name == "graphsage_dgi":
        generator = GraphSAGENodeGenerator(
            train_graph, batch_size=50, num_samples=[5])
        embedding_layer = GraphSAGE(
            layer_sizes=[params["embedding_dimension"]], activations=["relu"],
            generator=generator
        )
    else:
        raise ValueError(f"Unknown mode name {model_name}")

    embedding_model = _execute_deep_graph_infomax(
        train_graph, embedding_layer, generator, params)

    # Here the models can be both inductive and transductive
    if model_name in ["gcn_dgi", "gat_dgi", "graphsage_dgi"]:
        return embedding_model.predict(
            generator.flow(train_graph.nodes()))
    else:
        return embedding_model
