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
import random
import numpy as np
import pandas as pd
from nltk.corpus import words

from bluegraph.core.io import PandasPGFrame
from bluegraph.preprocess import ScikitLearnPGEncoder


def test_encoding(random_pgframe):
    edges_to_include = [
        (s, t)
        for s, t in random_pgframe.edges()
        if random_pgframe.get_edge(s, t)["mi"] != 0
    ]
    random_pgframe = random_pgframe.subgraph(edges=edges_to_include)

    N = random_pgframe.number_of_nodes()
    types = ["Apple", "Orange", "Carrot"]
    node_types = {
        n: np.random.choice(types, p=[0.5, 0.4, 0.1])
        for n in range(N)
    }
    random_pgframe.add_node_types(node_types)

    types = ["isFriend", "isEnemy"]
    edge_types = {
        e: np.random.choice(types, p=[0.8, 0.2])
        for e in random_pgframe.edges()
    }
    random_pgframe.add_edge_types(edge_types)
    # wegiht numeric, mi distance numeric

    colors = ["red", "green", "blue"]
    colors = pd.DataFrame(
        [
            (n, np.random.choice(colors))
            for n in random_pgframe.nodes()
        ],
        columns=["@id", "color"]
    )
    random_pgframe.add_node_properties(colors, prop_type="category")

    desc = pd.DataFrame(
        [
            (n, ' '.join(random.sample(words.words(), 20)))
            for n in random_pgframe.nodes()
        ],
        columns=["@id", "desc"]
    )

    random_pgframe.add_node_properties(desc, prop_type="text")

    shapes = ["dashed", "dotted", "solid"]
    shapes = pd.DataFrame(
        [
            (s, t, np.random.choice(shapes))
            for s, t, in random_pgframe.edges()
        ],
        columns=["@source_id", "@target_id", "shapes"]
    )
    random_pgframe.add_edge_properties(shapes, prop_type="category")

    desc = pd.DataFrame(
        [
            (s, t, ' '.join(random.sample(words.words(), 20)))
            for s, t, in random_pgframe.edges()
        ],
        columns=["@source_id", "@target_id", "desc"]
    )
    random_pgframe.add_edge_properties(desc, prop_type="text")
    hom_encoder = ScikitLearnPGEncoder(
        node_properties=["weight", "color", "desc"],
        edge_properties=["mi", "distance", "shapes", "desc"],
        edge_features=True,
        heterogeneous=False,
        encode_types=True,
        drop_types=True,
        text_encoding="tfidf",
        standardize_numeric=True)
    transformed_frame = hom_encoder.fit_transform(random_pgframe)
    hom_encoder = ScikitLearnPGEncoder(
        node_properties={
            "Apple": ["weight", "color", "desc"],
            "Orange": ["color", "desc"],
            "Carrot": ["desc"]
        },
        edge_features=False,
        heterogeneous=True,
        encode_types=True,
        drop_types=True,
        text_encoding="word2vec",
        standardize_numeric=True)
    transformed_frame = hom_encoder.fit_transform(random_pgframe)
