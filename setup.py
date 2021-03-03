# Copyright (c) 2020–2021, EPFL/Blue Brain Project
#
# Blue Graph is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Blue Graph is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
# General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Blue Graph. If not, see <https://choosealicense.com/licenses/lgpl-3.0/>.

from setuptools import setup
import os

from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file.
with open(os.path.join(HERE, "README.rst"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="bluegraph",
    author="Blue Brain Project, EPFL",
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
        "kganalytics",
        "cord19kg",
        "cord19kg.apps"
    ],
    package_data={
        'cord19kg.apps': [
            'assets/*',
            'assets/fontawesome-5.15.1-web/',
            'assets/fontawesome-5.15.1-web/css/*',
            'assets/fontawesome-5.15.1-web/webfonts/*'
        ]
    },
    python_requires=">=3.6",
    setup_requires=[
        "setuptools_scm",
    ],
    install_requires=[
        "numpy>=1.16.5",
        "pandas",
        "networkx",
        "python-louvain"
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
            "nexusforge",
            "nexus-sdk",
            "pyjwt==1.7.1",
            "ipywidgets"
        ],
        "docs": ["sphinx", "sphinx-bluebrain-theme"],
    },
    classifiers=[
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3 :: Only",
        "Natural Language :: English",
    ]
)
