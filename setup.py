# BlueGraph: unifying Python framework for graph analytics and co-occurrence analysis. 

# Copyright 2020-2021 Blue Brain Project / EPFL

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))


# Get the long description from the README file.
with open(os.path.join(HERE, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bluegraph",
    author="Blue Brain Project, EPFL",
    version="0.2.0",
    use_scm_version={
        "write_to": "bluegraph/version.py",
        "write_to_template": "__version__ = '{version}'\n",
    },
    description="Knowledge Graphs analytics.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    keywords="framework knowledge graph data science",
    url="https://github.com/BlueBrain/BlueBrainGraph",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "matplotlib",
        "nltk",
        "nexusforge",
        "gensim",
        "tensorflow"
    ],
    dependency_links=['http://github.com/user/repo/tarball/master#egg=package-1.0'],
    package_data={
        'cord19kg.apps': [
            'assets/*',
            'assets/fontawesome-5.15.1-web/',
            'assets/fontawesome-5.15.1-web/css/*',
            'assets/fontawesome-5.15.1-web/webfonts/*'
        ]
    },
    extras_require={
        "dev": [
            "tox", "pytest", "pytest-bdd", "pytest-cov==2.10.1",
            "pytest-mock==3.3.1", "codecov"
        ],
        "docs": ["sphinx", "sphinx-bluebrain-theme"],
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
        "networkx": [
            "networkx",
            "python-louvain"
        ],
        "graph-tool": [
            "graph-tool"
        ],
        "neo4j": [
            "neo4j"
        ],
        "stellargraph": [
            "stellargraph"
        ],
        "all": [
            "jupyter_dash",
            "dash_bootstrap_components",
            "dash_daq",
            "dash_extensions",
            "dash_cytoscape",
            "nexus-sdk",
            "pyjwt==1.7.1",
            "ipywidgets",
            "networkx",
            "python-louvain",
            "graph-tool",
            "neo4j",
            "stellargraph"
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
