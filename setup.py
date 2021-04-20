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
    use_scm_version={
        "write_to": "bluegraph/version.py",
        "write_to_template": "__version__ = '{version}'\n",
    },
    description="Knowledge Graphs analytics.",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    keywords="framework knowledge graph data science",
    url="https://github.com/BlueBrain/BlueGraph",
    packages=find_packages(),
    python_requires=">=3.7",
    setup_requires=[
        "setuptools_scm",
    ],
    install_requires=[
        "numpy>=1.20.1",
        "pandas",
        "sklearn",
        "scipy",
        "matplotlib",
        "nltk",
        "nexusforge @ git+https://github.com/BlueBrain/nexus-forge.git#nexusforge",
        "gensim==3.8.3",
        "tensorflow",
        "networkx>=2",
        "python-louvain"
    ],
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
            "pytest-mock==3.3.1", "codecov",
            "dash",
            "jupyter_dash",
            "dash_bootstrap_components",
            "dash_daq",
            "dash_extensions",
            "dash_cytoscape",
            "nexus-sdk",
            "pyjwt==1.7.1",
            "ipywidgets",
            "neo4j",
            "stellargraph"
        ],
        "docs": [
            "sphinx", "sphinx-bluebrain-theme",
            "dash",
            "jupyter_dash"  # a temporary solution, mocking this module fails
        ],
        "cord19kg": [
            "jupyter_dash",
            "dash_bootstrap_components",
            "dash_daq",
            "dash_extensions",
            "dash_cytoscape",
            "nexusforge @ git+https://github.com/BlueBrain/nexus-forge.git#nexusforge",
            "nexus-sdk",
            "pyjwt==1.7.1",
            "ipywidgets"
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
