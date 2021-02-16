class NodeClassifier(object):
    """A minimal wrapper around a classification model.

    This wrapper alows to build classification models of PGFrame nodes.
    """

    def __init__(self, model, feature_vector_prop=None, feature_props=None):
        self.model = model
        self.feature_vector_prop = feature_vector_prop
        self.feature_props = feature_props

    def _concatenate_feature_props(self, pgframe, nodes):
        if self.feature_props is None or len(self.feature_props) == 0:
            raise ValueError
        return pgframe.nodes(
            raw_frame=True).loc[nodes, self.feature_props].to_numpy()

    def fit(self, pgframe, train_nodes=None, labels=None, label_prop=None):
        # If no train nodes provided, use all nodes of the input graph
        if train_nodes is None:
            train_nodes = pgframe.nodes()

        # If no labels provided, try to use a label property
        if labels is None:
            if label_prop not in pgframe.node_properties():
                raise ValueError()
            labels = pgframe.get_node_property_values(
                label_prop).loc[train_nodes].tolist()

        # If no feature vector property provided,
        # try to concatenate feature_props
        if self.feature_vector_prop:
            data = pgframe.get_node_property_values(
                self.feature_vector_prop).loc[train_nodes].tolist()
        else:
            data = self._concatenate_feature_props(pgframe, train_nodes)
        self.model.fit(data, labels)

    def predict(self, pgframe, predict_nodes=None):
        # If no prediction nodes provided, use all nodes of the input graph
        if predict_nodes is None:
            predict_nodes = pgframe.nodes()
        if self.feature_vector_prop:
            data = pgframe.get_node_property_values(
                self.feature_vector_prop).loc[predict_nodes].tolist()
        else:
            data = self._concatenate_feature_props(pgframe, predict_nodes)
        return self.model.predict(data)
