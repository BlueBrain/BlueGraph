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
from itertools import chain

import pandas as pd

import pytest

from cord19kg.utils import mentions_to_occurrence, is_experiment_related, clean_up_entity, has_min_length
from kganalytics.network_generation import generate_cooccurrence_network

@pytest.fixture(scope="session")
def mentions_file_path():
    return os.sep.join((os.path.abspath("."), "tests/data/mentions_data_sample_1000.csv"))

@pytest.fixture(scope="session")
def mentions(mentions_file_path):
    mentions = pd.read_csv(mentions_file_path, nrows=1000)
    # Extract unique paper/section/paragraph identifiers
    mentions["paper"] = mentions["paper_id"].apply(
        lambda x: x.split(":")[0])
    mentions["section"] = mentions["paper_id"].apply(
        lambda x: ":".join([x.split(":")[0], x.split(":")[1]]))

    mentions = mentions.rename(columns={"paper_id": "paragraph"})
    return mentions

@pytest.fixture(scope="session")
def occurrence_data(mentions):
    # Occurence data
    occurrence_data, counts = mentions_to_occurrence(
        mentions,
        term_column="entity",
        factor_columns=["paper", "section", "paragraph"],
        term_cleanup=clean_up_entity,
        term_filter=lambda x: has_min_length(x, 2),
        mention_filter=lambda data: ~data["section"].apply(is_experiment_related),
        dump_prefix="tests/data/example_")

    # Filter entities that occur only once (only in one paragraph, usually represent noisy terms)
    occurrence_data = occurrence_data[occurrence_data["paragraph"].apply(lambda x: len(x) > 1)]
    assert counts == {'paper': 38, 'section': 287, 'paragraph': 504}
    assert len(occurrence_data) == 131
    return occurrence_data, counts

@pytest.fixture(scope="session")
def paper_comention_network_100_most_frequent(occurrence_data):

    # Load 10000 lines of the mention data sample


    # Use only 100 most frequent entities
    paper_comention_network_100_most_frequent = generate_cooccurrence_network(
        occurrence_data[0], "paper", occurrence_data[1]["paper"],
        n_most_frequent=100,
        parallelize=False)

    assert paper_comention_network_100_most_frequent is not None
    assert len(paper_comention_network_100_most_frequent) == 100

    edge_attributes = set(chain.from_iterable(d.keys() for *_, d in paper_comention_network_100_most_frequent.edges(data=True)))

    assert edge_attributes =={
        "frequency",
        "ppmi",
        "npmi",
        "distance_ppmi",
        "distance_npmi"
    }

    return paper_comention_network_100_most_frequent

@pytest.fixture(scope="session")
def paper_comention_network_1000_edges(occurrence_data):

    # Limit to 1000 edges
    paper_comention_network_1000_edges = generate_cooccurrence_network(
        occurrence_data[0], "paper", occurrence_data[1]["paper"],
        limit=1000,
        parallelize=False
    )

    return paper_comention_network_1000_edges