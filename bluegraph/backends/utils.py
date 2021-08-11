"""A set of factory utils for different processing backends."""

from bluegraph.exceptions import BlueGraphException
from .configs import (ANALYZER_CLS, EMBEDDER_CLS)


def create_analyzer(analyzer_type, backend,
                    pgframe=None, directed=True,
                    uri=None, username=None, password=None,
                    driver=None, node_label=None, edge_label=None):
    """Create an analyzer interface for a given type and backend."""
    if analyzer_type not in ANALYZER_CLS:
        raise BlueGraphException(
            f"Analyzer type '{analyzer_type}' is not implemented, "
            "available analyzers are: " + ", ".join(
                [f"'{el}'" for el in ANALYZER_CLS.keys()])
        )
    if backend not in ANALYZER_CLS[analyzer_type]:
        verbose_analyzer_name = analyzer_type.replace("_", " ").capitalize()
        raise BlueGraphException(
            f"{verbose_analyzer_name} is not enabled or not implemented for "
            f"the backend '{backend}', available backends are: " + ", ".join(
                [f"'{el}'" for el in ANALYZER_CLS[analyzer_type].keys()])
        )
    cls = ANALYZER_CLS[analyzer_type][backend]
    return (
        cls(
            pgframe=pgframe, directed=directed,
            uri=uri, username=username, password=password,
            driver=driver, node_label=node_label, edge_label=edge_label)
        if backend == "neo4j"
        else cls(pgframe=pgframe, directed=directed)
    )


def create_node_embedder(backend, model_name,
                         directed=True, include_type=False,
                         feature_props=None, feature_vector_prop=None,
                         edge_weight=None, **model_params):
    """Create a node embedding interface for a given backend."""
    if backend not in EMBEDDER_CLS:
        raise BlueGraphException(
            f"Node embedder corresponding to the backend '{backend}' "
            "is not enabled or not implemented, available backends are: " +
            ", ".join([f"'{el}'" for el in EMBEDDER_CLS.keys()]))
    return EMBEDDER_CLS[backend](
        model_name=model_name, directed=directed, include_type=include_type,
        feature_props=feature_props, feature_vector_prop=feature_vector_prop,
        edge_weight=edge_weight, **model_params)
