from setuptools import setup


setup(
    name="bbg_analytics",
    author="Blue Brain Project, EPFL",
    version="0.0.1",
    packages=[
        "kganalytics",
        "cord_analytics",
        "bbg_apps",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "networkx",
        "python-louvain"
    ],
)
