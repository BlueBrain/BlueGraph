from setuptools import setup


setup(
    name="bbg_analytics",
    author="Blue Brain Project, EPFL",
    version="0.0.1",
    packages=[
        "kganalytics",
        "cord_19",
        "cord_19.apps"
    ],
    package_data={
        'cord_19.apps': ['assets/*'],
    },
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "networkx",
        "python-louvain",
        "jupyter_dash",
        "dash_bootstrap_components",
        "dash_daq"
    ],
)
