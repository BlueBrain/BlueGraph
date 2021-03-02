========================
:code:`cord19kg` package
========================

The package contains a set of tools for interactive exploration and analysis of the `CORD-19 <https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge>`_ dataset using the co-occurrence analysis of the extracted named entities. It includes data preparation and curation helpers, tools for generation and analysis of co-occurrence networks. Moreover, it provides several mini-applications (based on `JupyterDash <https://github.com/plotly/jupyter-dash>`_ and `ipywidgets <https://ipywidgets.readthedocs.io/en/stable/>`_) for Jupyter notebooks allowing the user to interactively perform:

- entity curation;
- network visualization and analysis;
- dataset saving/loading from `Nexus <https://bluebrainnexus.io/>`_.


Interactive applications
------------------------


Examples and tutorials
----------------------

The :code:`cord19kg` packages provides `examples <https://github.com/BlueBrain/BlueBrainGraph/tree/refactoring/cord19kg/examples>`_ of CORD-19-specific co-occurrence analysis. We recommend starting from the `Co-occurrence analysis tutorial <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Co-occurrence%20analysis%20tutorial.ipynb>`_ notebook providing a simple starting example (including a small guide to the visualization app interface).

The `Topic-centered co-occurrence network analysis of CORD-19 <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Glucose%20is%20a%20risk%20facor%20for%20COVID-19%20(3000%20papers).ipynb>`_ notebook provides a full analysis pipeline on the selection of 3000 articles obtained by searching the CORD-19 dataset using the query *"Glucose is a risk factor for COVID-19"* (the search is performed using `BlueBrainSearch <https://github.com/BlueBrain/BlueBrainSearch>`_).

The `Nexus-hosted co-occurrence network analysis of CORD-19 <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Nexus-hosted%20topic-centered%20analysis%20(3000%20papers).ipynb>`_ notebook provides an example for the previously mentioned 3000-article dataset, where datasets corresponding to different analysis steps can be saved and loaded to and from a `Blue Brain Nexus <https://bluebrainnexus.io/>`_ project.

Finally, the :code:`generate_10000_networks.py` `script <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/generate_10000_network.py>`_ allows the user to generate the co-occurrence networks for 10'000 most frequent entities extracted from the entire CORD-19v47 database (based on paper- and paragraph- level entity co-occurrence). To run the script, simply execute :code:`python generate_10000_networks.py` from the examples folder.

Note that the generated networks are highly dense (contain a large number of edges, for example, ~44M edges for the paper-based network), and the process of their generation, even if parallelized, is highly costly.
