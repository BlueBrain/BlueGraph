"""Backend configs and mappings."""
from .networkx import (NXMetricProcessor,
                       NXPathFinder,
                       NXCommunityDetector)

try:
    from .graph_tool import (GTMetricProcessor,
                             GTPathFinder,
                             GTCommunityDetector)
    DISABLED_GT = False
except ImportError:
    DISABLED_GT = True

try:
    from .neo4j import (Neo4jMetricProcessor,
                        Neo4jPathFinder,
                        Neo4jCommunityDetector,
                        Neo4jNodeEmbedder)
    DISABLED_NEO4J = False
except ImportError:
    DISABLED_NEO4J = True

try:
    from .stellargraph import StellarGraphNodeEmbedder
    DISABLED_STELLAR = False
except ImportError:
    DISABLED_STELLAR = True


# -----------------------------------------------------------
# Interface mappings for different analytics backends

ANALYZER_CLS = {
    "metrics_processor": {
        "networkx": NXMetricProcessor,
    },
    "path_finder": {
        "networkx": NXPathFinder,
    },
    "community_detector": {
        "networkx": NXCommunityDetector,
    }
}

EMBEDDER_CLS = {}


if not DISABLED_GT:
    ANALYZER_CLS["metrics_processor"]["graph_tool"] = GTMetricProcessor
    ANALYZER_CLS["path_finder"]["graph_tool"] = GTPathFinder
    ANALYZER_CLS["community_detector"]["graph_tool"] = GTCommunityDetector

if not DISABLED_NEO4J:
    ANALYZER_CLS["metrics_processor"]["neo4j"] = Neo4jMetricProcessor
    ANALYZER_CLS["path_finder"]["neo4j"] = Neo4jPathFinder
    ANALYZER_CLS["community_detector"]["neo4j"] = Neo4jCommunityDetector
    EMBEDDER_CLS["neo4j"] = Neo4jNodeEmbedder


if not DISABLED_STELLAR:
    EMBEDDER_CLS["stellargraph"] = StellarGraphNodeEmbedder
