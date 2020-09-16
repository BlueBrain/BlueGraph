"""Exampel of network analysis pipeline from a raw mention data."""
import pandas as pd

from analytics.generate_network import generate_comention_network
from analytics.data_preparation import (mentions_to_occurrence,
                                        is_experiment_related,
                                        dummy_clean_up,
                                        is_not_single_letter)


if __name__ == '__main__':
    print("Reading mentions data...")
    mentions = pd.read_csv("data/mention_data_sample_50000.csv")

    # Extract unique paper/seciton/paragraph identifiers
    mentions["paper"] = mentions["paper_id"].apply(
        lambda x: x.split(":")[0])
    mentions["section"] = mentions["paper_id"].apply(
        lambda x: ":".join([x.split(":")[0], x.split(":")[1]]))

    mentions = mentions.rename(columns={"paper_id": "paragraph"})

    print("Extracting occurrences...")
    occurrence_data = mentions_to_occurrence(
        mentions,
        term_column="entity",
        factor_columns=["paper", "section", "paragraph"],
        term_cleanup=dummy_clean_up,
        term_filter=is_not_single_letter,
        mention_filter=lambda data: ~data["section"].apply(is_experiment_related),
        dump_prefix="data/occurrence_data.csv")

    print("Generating a network...")
    # Limit to 1000 edges
    comention_network = generate_comention_network(
        occurrence_data,
        factor_column="paper",
        limit=1000,
        parallelize=True)
