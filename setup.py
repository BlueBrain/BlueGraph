from setuptools import setup


setup(
    name="bbg_analytics",
    author="Blue Brain Project, EPFL",
    packages=[
        "kganalytics",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "networkx",
        "python-louvain"
    ],
)
