# This script is distributed with the 3-clause BSD license.

# See file AUTHORS.rst for further details.

# COPYRIGHT 2020–2021, Blue Brain Project/EPFL

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.

#  * Neither the name of the NetworkX Developers nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

"""Generate 10'000 entity co-occurrence networks from CORD-19.

This script generates two co-occurrence networks: for paper- and
paragraph-based entity co-occurence. The input data for the
generation process is a table whose rows
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

- the paper-based network: `Top_100000_network_paper_graph.json`;

- the minimum spanning tree of the paper-based network:
`Top_100000_network_paper_tree.json`;

- the paragraph-based network: `Top_100000_network_paragraph_graph.json`;

- the minimum spanning tree of the paragraph-based network:
`Top_100000_network_paragraph_tree.json`;

To open the stored graphs try `PandasPGFrame.load_json(<graph_json_file>)` to
open the stored graphs as PandasPGFrame graph objects.

"""
import json
import pandas as pd
import time
import zipfile

from pathlib import Path

from cord19kg.utils import generate_cooccurrence_analysis, download_from_nexus

if __name__ == '__main__':
    # Load the input data
    start = time.time()
    print("Loading the input data...")
    nexus_bucket = "covid19-kg/data"
    nexus_endpoint = "https://bbp.epfl.ch/nexus/v1"
    download_from_nexus(uri=f"{nexus_endpoint}/resources/{nexus_bucket}/_/2a3c1698-3881-4022-8439-3a474635ec86",
                        output_path="data",
                        config_file_path="./config/data-download-nexus.yml",
                        nexus_endpoint=nexus_endpoint,
                        nexus_bucket=nexus_bucket, unzip=True)
    print("\tLoading the occurrence data file...")
    data = pd.read_json("data/CORD_19_v47_occurrence_top_10000.json")
    print("\tPreprocessing the occurrence data file...")
    data["paper"] = data["paper"].apply(set)
    data["paragraph"] = data["paragraph"].apply(set)
    factor_counts_metadata = download_from_nexus(
        uri=f"{nexus_endpoint}/resources/{nexus_bucket}/_/52471b18-5761-4884-867c-1afc81787d4b",
        output_path="data",
        nexus_endpoint=nexus_endpoint,
        nexus_bucket=nexus_bucket)
    with open(f"data/{factor_counts_metadata.distribution.name}", "r") as f:
        factor_counts = json.load(f)
    print("Done in {:.2f}s.".format(time.time() - start))

    # Create the output folder if doesn't exist
    Path("data/output_graphs").mkdir(parents=True, exist_ok=True)

    backend_configs = {
        "metrics": "graph_tool",
        "communities": "networkx",
        "paths": "graph_tool"
    }

    # Generate graphs and run the analysis pipeline
    print("Generating co-occurrence networks...")
    start = time.time()
    graphs, trees = generate_cooccurrence_analysis(
        data, factor_counts,
        type_data=data[["entity_type"]].reset_index().rename(
            columns={"index": "entity", "entity_type": "type"}),
        n_most_frequent=10000,
        factors=["paper", "paragraph"],
        cores=8,  # Change the number of cores if necessary
        graph_dump_prefix="data/output_graphs/Top_100000_network",
        communities=True,
        remove_zero_mi=True,
        backend_configs=backend_configs)
    print("Done in {:.2f}s.".format(time.time() - start))
