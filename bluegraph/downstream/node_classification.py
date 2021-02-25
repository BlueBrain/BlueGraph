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
