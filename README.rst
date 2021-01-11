****************
Blue Brain Graph
****************

Toolkit for graph analytics and co-occurrence analysis

About
#########################

This repository contains a set of packages that bring together various graph analytics tools. The current implementation is based on the `NetworkX` library.

The `kganalytics` package
*************************

The package gathers a set of tools that allow

- generating co-occurrence networks;
- computing mutual-information-based co-occurrence scores;
- computing centrality measures;
- detecting node communities;
- finding spanning trees;
- computing (sets of) weighted shortest paths.


The `cord19kg` package
**********************

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

To get started and get familiar with the ideas behind the co-occurrence analysis and graph analytics, in general, the `kganalytics`_ package provides two example notebooks for `literature exploration <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/kganalytics/notebooks/Literature%20exploration.ipynb>`_ and `NASA dataset keywords exploration <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/kganalytics/notebooks/NASA%20dataset%20keywords.ipynb>`_.

`examples <https://github.com/BlueBrain/BlueBrainGraph/tree/refactoring/cord19kg/examples>`_

`Co-occurrence analysis tutorial <https://github.com/BlueBrain/BlueBrainGraph/blob/refactoring/cord19kg/examples/notebooks/Co-occurrence%20analysis%20tutorial.ipynb>`_




Aknowledgements
###############

This project has received funding from the EPFL Blue Brain Project (funded by the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology).
