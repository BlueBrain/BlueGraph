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
from collections import namedtuple
import warnings
import pandas as pd

from gensim.models.poincare import PoincareModel

from bluegraph.core.embed.embedders import GraphElementEmbedder
from bluegraph.backends.params import (GENSIM_PARAMS,
                                       DEFAULT_GENSIM_PARAMS)


GensimGraph = namedtuple('GensimGraph', 'graph graph_configs')


class GensimNodeEmbedder(GraphElementEmbedder):

    _transductive_models = [
        "poincare",
        "word2vec"
    ]

    def __init__(self, model_name, directed=True, include_type=False,
                 feature_props=None, feature_vector_prop=None,
                 edge_weight=None, **model_params):
        if directed is False and model_name == "poincare":
            raise GraphElementEmbedder.FittingException(
                "Poincare embedding can be performed only on directed graphs: "
                "undirected graph was provided")
        super().__init__(
            model_name=model_name, directed=directed,
            include_type=include_type,
            feature_props=feature_props,
            feature_vector_prop=feature_vector_prop,
            edge_weight=edge_weight, **model_params)

    @staticmethod
    def _generate_graph(pgframe, graph_configs):
        """Generate backend-specific graph object."""
        return GensimGraph(pgframe, graph_configs)

    def _dispatch_model_params(self, **kwargs):
        """Dispatch training parameters."""
        params = {}
        for k, v in kwargs.items():
            if k not in GENSIM_PARAMS[self.model_name]:
                warnings.warn(
                    f"GensimNodeEmbedder's model '{self.model_name}' "
                    f"does not support the training parameter '{k}', "
                    "the parameter will be ignored",
                    GraphElementEmbedder.FittingWarning)
            else:
                params[k] = v

        for k, v in DEFAULT_GENSIM_PARAMS.items():
            if k not in params:
                params[k] = v
        return params

    def _fit_transductive_embedder(self, train_graph):
        """Fit transductive embedder (no model, just embeddings)."""

        model_params = {**self.params}
        del model_params["epochs"]

        if self.model_name == "poincare":
            model = PoincareModel(
                train_graph.graph.edges(), **model_params)

        model.train(epochs=self.params["epochs"])

        embedding = pd.DataFrame(
            [
                (n, model.kv.get_vector(n))
                for n in train_graph.graph.nodes()
            ],
            columns=["@id", "embedding"]
        ).set_index("@id")
        return embedding

    def _fit_inductive_embedder(self, train_graph):
        """Fit inductive embedder (predictive model and embeddings)."""
        raise NotImplementedError(
            "Inductive models are not implemented for gensim-based "
            "node embedders")

    def _predict_embeddings(self, graph, nodes=None):
        """Fit inductive embedder (predictive model and embeddings)."""
        raise NotImplementedError(
            "Inductive models are not implemented for gensim-based "
            "node embedders")

    @staticmethod
    def _save_predictive_model(model, path):
        pass

    @staticmethod
    def _load_predictive_model(path):
        pass
