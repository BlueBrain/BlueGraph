****************
Blue Brain Graph
****************

Toolkit for graph analytics and co-occurrence analysis

About
#########################

This repository contains a set of packages that bring together various graph analytics tools. The current implementation is based on the `NetworkX <https://networkx.org/>`_ library.

The :code:`kganalytics` package
*******************************

The package gathers a set of tools that allow

- generating co-occurrence networks;
- computing mutual-information-based co-occurrence scores;
- computing centrality measures;
- detecting node communities;
- finding spanning trees;
- computing (sets of) weighted shortest paths.


The :code:`cord19kg` package
****************************

The package contains a set of tools for interactive exploration and analysis of the `CORD-19 <https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge>`_ dataset using the co-occurrence analysis of the extracted named entities. It includes data preparation and curation helpers, tools for generation and analysis of co-occurrence networks. Moreover, it provides several mini-applications (based on `JupyterDash <https://github.com/plotly/jupyter-dash>`_ and `ipywidgets <https://ipywidgets.readthedocs.io/en/stable/>`_) for Jupyter notebooks allowing the user to interactively perform:

- entity curation;
- network visualization and analysis;
- dataset saving/loading from `Nexus <https://bluebrainnexus.io/>`_.

Installation
############

It is recommended to use a virtual environment such as `venv <https://docs.python.org/3.6/library/venv.html>`_  or `conda environment <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

Development version can be installed using:

::

  pip install git+https://github.com/BlueBrain/BlueBrainGraph.git

Getting started
###############

To get started and get familiar with the ideas behind the co-occurrence analysis and graph analytics, in general, the :code:`kganalytics` package provides two example notebooks: `Literature exploration <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/kganalytics/notebooks/Literature%20exploration.ipynb>`_ and `NASA dataset keywords exploration <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/kganalytics/notebooks/NASA%20dataset%20keywords.ipynb>`_.

The :code:`cord19kg` packages provides `examples <https://github.com/BlueBrain/BlueBrainGraph/tree/refactoring/cord19kg/examples>`_ of CORD-19-specific co-occurrence analysis. We recommend starting from the `Co-occurrence analysis tutorial <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Co-occurrence%20analysis%20tutorial.ipynb>`_ notebook providing a simple starting example.

The `Topic-centered co-occurrence network analysis of CORD-19 <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Glucose%20is%20a%20risk%20facor%20for%20COVID-19%20(3000%20papers).ipynb>`_ notebook provides a full analysis pipeline on the selection of 3000 articles obtained by searching the CORD-19 dataset using the query *"Glucose is a risk factor for COVID-19"* (the search is performed using `BlueBrainSearch <https://github.com/BlueBrain/BlueBrainSearch>`_).

Finally, the `Nexus-hosted co-occurrence network analysis of CORD-19 <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Nexus-hosted%20topic-centered%20analysis%20(3000%20papers).ipynb>`_ notebook provides an example for the previously mentioned 3000-article dataset, where datasets corresponding to different analysis steps can be saved and loaded to and from a `Blue Brain Nexus <https://bluebrainnexus.io/>`_ project.

Aknowledgements
###############

This project has received funding from the EPFL Blue Brain Project (funded by the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology).
