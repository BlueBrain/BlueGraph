"""Generate 10'000 entity co-occurrence networks from CORD-19.

This script generates two co-occurrence networks: for paper- and
paragraph-based entity co-occurence. The input data for the
generation process is a table (the compressed table is provided in
`./data/CORD_19_v47_occurrence_top_10000.json.zip`) whose rows
correspond to unique entities (linked to NCIT ontology terms when possible).
Its columns contain:

- types of these entities (either resolved from
the NCIT class hierarchy or computed from raw entity types given
by the NER model)
- occurrence of the entities in sets of papers and paragraphs.

The output of the script is saved in the directory `./data/output_graphs`:
networks' nodes and edges (together with their attributes) are stored as
pickled pandas dataframes.

The following graphs are stored in the output folder:

- the paper-based network: `Top_100000_network_paper_node_list.pkl`
and `Top_100000_network_paper_edge_list.pkl`;

- the minimum spanning tree of the paper-based network:
`Top_100000_network_paper_tree_node_list.pkl` and
`Top_100000_network_paper_tree_edge_list.pkl`;

- the paragraph-based network:
`Top_100000_network_paragraph_node_list.pkl`
and `Top_100000_network_paper_edge_list.pkl`;

- the minimum spanning tree of the paragraph-based network:
`Top_100000_network_paper_paragraph_node_list.pkl` and
`Top_100000_network_paper_paragraph_edge_list.pkl`;

Try `kganalytics.export.load_network` to open the stored graphs as
NetworkX graph objects.

"""
import json
import pandas as pd
import time
import zipfile

from pathlib import Path

from cord19kg.utils import generate_cooccurrence_analysis


if __name__ == '__main__':
    # Load the input data
    start = time.time()
    print("Loading the input data...")
    print("\tDecompressing the occurrence data file...")
    with zipfile.ZipFile(
            "data/CORD_19_v47_occurrence_top_10000.json.zip", 'r') as zip_ref:
        zip_ref.extractall("data/")
    print("\tLoading the occurrence data file...")
    data = pd.read_json("data/CORD_19_v47_occurrence_top_10000.json")
    print("\tPreprocessing the occurrence data file...")
    data["paper"] = data["paper"].apply(set)
    data["paragraph"] = data["paragraph"].apply(set)
    with open("data/CORD_19_v47_factor_counts.json", "r") as f:
        factor_counts = json.load(f)
    print("Done in {:.2f}s.".format(time.time() - start))

    # Create the output folder if doesn't exist
    Path("data/output_graphs").mkdir(parents=True, exist_ok=True)

    # Generate graphs and run the analysis pipeline
    print("Generating co-occurrence networks...")
    start = time.time()
    graphs, trees = generate_cooccurrence_analysis(
        data, factor_counts,
        type_data=data[["entity_type"]].rename(
            columns={"entity_type": "type"}),
        n_most_frequent=10000,
        factors=["paper", "paragraph"],
        cores=8,  # Change the number of cores if necessary
        graph_dump_prefix="data/output_graphs/Top_100000_network",
        communities=True,
        remove_zero_mi=True)
    print("Done in {:.2f}s.".format(time.time() - start))
