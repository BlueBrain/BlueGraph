from abc import ABC, abstractmethod

import numpy as np


class ElementClassifier(ABC):
    """Interface for graph element classification models.

    It wraps a predictive classification model provided by the user
    and a set of configs that allow the user to fit the model
    and make predictions on the input PGFrames. Its main goal is to
    hide the details on converting element (node or edge) properties
    into data tables that can be provided to the predictive model.
    """
    def __init__(self, model, feature_vector_prop=None, feature_props=None,
                 **kwargs):
        self.model = model
        self.feature_vector_prop = feature_vector_prop
        self.feature_props = feature_props

    def _concatenate_feature_props(self, pgframe, nodes):
        if self.feature_props is None or len(self.feature_props) == 0:
            raise ValueError
        return pgframe.nodes(
            raw_frame=True).loc[nodes, self.feature_props].to_numpy()

    def _get_node_features(self, pgframe, nodes):
        if self.feature_vector_prop:
            features = pgframe.get_node_property_values(
                self.feature_vector_prop, nodes=nodes).tolist()
        else:
            features = self._concatenate_feature_props(pgframe, nodes)
        return np.array(features)

    @abstractmethod
    def _generate_train_elements(self, pgfame, elements=None):
        pass

    @abstractmethod
    def _generate_predict_elements(self, pgfame, elements=None):
        pass

    @abstractmethod
    def _generate_train_labels(self, pgframe, elements, label_prop=None):
        pass

    @abstractmethod
    def _generate_data_table(self, pgframe, elements):
        pass

    def fit(self, pgframe, train_elements=None, labels=None, label_prop=None,
            **kwargs):
        train_elements = self._generate_train_elements(
            pgframe, train_elements, **kwargs)
        labels = self._generate_train_labels(
            pgframe, train_elements, label_prop) if labels is None else labels
        data = self._generate_data_table(pgframe, train_elements)
        self.model.fit(data, labels)

    def predict(self, pgframe, predict_elements=None):
        predict_elements = self._generate_predict_elements(
            pgframe, predict_elements)
        data = self._generate_data_table(pgframe, predict_elements)
        return self.model.predict(data)
