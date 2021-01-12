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

from setuptools import setup

setup(
    name="bbg_analytics",
    author="Blue Brain Project, EPFL",
    version="0.0.1",
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
        ],
    },
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "networkx",
        "python-louvain",
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
)
