Getting Started with bluegraph
------------------------------


The `examples directory <https://github.com/BlueBrain/BlueGraph/tree/master/examples>`_ contains a set of Jupyter notebooks providing tutorials and usecases for BlueGraph.

To get started with property graph data structure `PGFrame` provided by BlueGraph, get an example of semantic property encoding, see :ref:`intro_pgframe_tutorial` (`notebook <https://github.com/BlueBrain/BlueGraph/blob/master/examples/notebooks/PGFrames%20and%20sematic%20encoding%20tutorial.ipynb>`__).

To get familiar with the ideas behind the co-occurrence analysis and the graph analytics interface provided by BlueGraph we recommend to have a look at the following tutorials: 

- :ref:`literature_tutorial` illustrates how to use BlueGraphs's analytics API for in-memory graph backends based on the `NetworkX` and the `graph-tool` libraries (`notebook <https://github.com/BlueBrain/BlueGraph/blob/master/examples/notebooks/Literature%20exploration%20(PGFrames%20%2B%20in-memory%20analytics%20tutorial).ipynb>`__).

- :ref:`nasa_tutorial` illustrates how to use the Neo4j-based analytics API for persistent property graphs (`notebook <https://github.com/BlueBrain/BlueGraph/blob/master/examples/notebooks/NASA%20keywords%20(PGFrames%20%2B%20Neo4j%20analytics%20tutorial).ipynb>`__).

- :ref:`embedding_tutorial` starts from the co-occurrence graph generation example and guides the user through the graph representation learning and all it's downstream tasks including node similarity queries, node classification, edge prediction and embedding pipeline building (`notebook <https://github.com/BlueBrain/BlueGraph/blob/master/examples/notebooks/Embedding%20and%20downstream%20tasks%20tutorial.ipynb>`__).
