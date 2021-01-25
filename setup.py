#
# Blue Brain Graph is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Blue Brain Graph is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Blue Brain Graph. If not, see <https://choosealicense.com/licenses/lgpl-3.0/>.
import os
from setuptools import setup

HERE = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file.
with open(os.path.join(HERE, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bluegraph",
    author="Blue Brain Project, EPFL",
    version="0.2.0",
    use_scm_version={
        "write_to": "kganalytics/version.py",
        "write_to_template": "__version__ = '{version}'\n",
    },
    description="Knowledge Graphs analytics.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    keywords="framework knowledge graph data science",
    url="https://github.com/BlueBrain/BlueBrainGraph",
    packages=[
        "bluegraph",
        "bluegraph.core",
        "bluegraph.core.analyse",
        "bluegraph.core.embed",
        "bluegraph.preprocess",
        "bluegraph.backends",
        "bluegraph.backends.networkx",
        "bluegraph.backends.networkx.analyse",
        "bluegraph.backends.graph_tool",
        "bluegraph.backends.graph_tool.analyse",
        "bluegraph.backends.neo4j",
        "bluegraph.backends.neo4j.analyse",
        "bluegraph.backends.stellargraph",
        "bluegraph.backends.stellargraph.embed",
        "bluegraph.downstream"
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "matplotlib",
        "nltk",
        "nexusforge"
    ],
    extras_require={
        "dev": [
            "tox", "pytest", "pytest-bdd", "pytest-cov==2.10.1",
            "pytest-mock==3.3.1", "codecov"
        ],
        "cord19kg": [
            "jupyter_dash",
            "dash_bootstrap_components",
            "dash_daq",
            "dash_extensions",
            "dash_cytoscape",
            "nexus-sdk",
            "pyjwt==1.7.1",
            "ipywidgets"
        ],
        "docs": ["sphinx", "sphinx-bluebrain-theme"],
        "networkx": [
            "networkx",
            "python-louvain"
        ],
        "graph_tool": [
            "graph-tool"
        ],
        "neo4j": [
            "neo4j"
        ],
        "stellargraph": [
            "stellargraph",
            "tensorflow"
        ]
    },
    classifiers=[
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English",
    ]
)
