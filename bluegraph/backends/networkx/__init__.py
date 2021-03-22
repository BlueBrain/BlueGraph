"""Imports to build an API.

The following usage is desirable:
from blugraph.networkx import NXMetricProcessor
from blugraph.networkx import NXPathFinder
"""
from .analyse.metrics import NXMetricProcessor
from .analyse.paths import NXPathFinder
from .analyse.communities import NXCommunityDetector

from .io import (NXGraphProcessor, pgframe_to_networkx, networkx_to_pgframe)
