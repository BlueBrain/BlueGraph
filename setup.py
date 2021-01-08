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
        "dash_cytoscape"
    ],
)
