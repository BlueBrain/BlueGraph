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
from .data_structures import ElementClassifier


class NodeClassifier(ElementClassifier):
    """Interface for node classification models.

    This wrapper alows to build classification models of PGFrame nodes.
    """
    def _generate_train_elements(self, pgframe, elements=None):
        if elements is None:
            elements = pgframe.nodes()
        return elements

    def _generate_predict_elements(self, pgframe, elements=None):
        if elements is None:
            elements = pgframe.nodes()
        return elements

    def _generate_train_labels(self, pgframe, elements, label_prop=None):
        if label_prop not in pgframe.node_properties():
            raise ValueError()
        labels = pgframe.get_node_property_values(
            label_prop, nodes=elements).tolist()
        return labels

    def _generate_data_table(self, pgframe, elements):
        return self._get_node_features(pgframe, elements)
