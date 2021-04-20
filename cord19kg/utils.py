# BlueGraph: unifying Python framework for graph analytics and co-occurrence analysis. 

# Copyright 2020-2021 Blue Brain Project / EPFL

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Module containing utils for COVID-19 network generation and analysis."""
import ast
import math
import operator
import pickle
import zipfile

import pandas as pd
import networkx as nx

from collections import Counter

from kgforge.core import KnowledgeGraphForge
from networkx.readwrite.json_graph.cytoscape import cytoscape_data

from bluegraph.core.io import PandasPGFrame
from bluegraph.preprocess import CooccurrenceGenerator
from bluegraph.backends.networkx import (NXMetricProcessor,
                                         NXCommunityDetector,
                                         NXPathFinder,
                                         NXGraphProcessor,
                                         networkx_to_pgframe,
                                         pgframe_to_networkx)
try:
    from bluegraph.backends.neo4j import (Neo4jMetricProcessor,
                                          Neo4jCommunityDetector,
                                          Neo4jPathFinder,
                                          Neo4jGraphProcessor,
                                          neo4j_to_pgframe,
                                          pgframe_to_neo4j)
    DISABLED_NEO4J = False
except ImportError:
    DISABLED_NEO4J = True
try:
    from bluegraph.backends.graph_tool import (GTMetricProcessor,
                                               GTCommunityDetector,
                                               GTPathFinder,
                                               GTGraphProcessor,
                                               graph_tool_to_pgframe,
                                               pgframe_to_graph_tool)
    DISABLED_GRAPH_TOOL = False
except ImportError:
    DISABLED_GRAPH_TOOL = True


BACKEND_MAPPING = {
    "networkx": {
        "metrics": NXMetricProcessor,
        "communities": NXCommunityDetector,
        "paths": NXPathFinder,
        "to_pgframe": networkx_to_pgframe,
        "from_pgframe": pgframe_to_networkx,
        "object_processor": NXGraphProcessor,
    }
}

if not DISABLED_NEO4J:
    BACKEND_MAPPING["neo4j"] = {
        "metrics": Neo4jMetricProcessor,
        "communities": Neo4jCommunityDetector,
        "paths": Neo4jPathFinder,
        "to_pgframe": neo4j_to_pgframe,
        "from_pgframe": pgframe_to_neo4j,
        "object_processor": Neo4jGraphProcessor,
    }

if not DISABLED_GRAPH_TOOL:
    BACKEND_MAPPING["graph_tool"] = {
        "metrics": GTMetricProcessor,
        "communities": GTCommunityDetector,
        "paths": GTPathFinder,
        "to_pgframe": graph_tool_to_pgframe,
        "from_pgframe": pgframe_to_graph_tool,
        "object_processor": GTGraphProcessor
    }


NON_ASCII_REPLACE = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "κ": "kappa",
    "’": "'",
    "–": "-",
    "‐": "-",
    "é": "e",
    "ó": "o"
}


def clean_up_entity(s):
    """Clean-up entity by removing common errors from NER."""
    s = str(s).lower()
    result = s.strip().strip("\"").strip("\'")\
        .strip("&").strip("#").replace(".", "").replace("- ", "-")

    # cleaning-up non-ascii symbols
    for s in result:
        try:
            s.encode('ascii')
        except UnicodeEncodeError:
            if s in NON_ASCII_REPLACE:
                s_ascii = NON_ASCII_REPLACE[s]
            else:
                s_ascii = ""
            result = result.replace(s, s_ascii)

    return result


def has_min_length(entities, length):
    """Check if a term has the min required length."""
    return entities.apply(lambda x: len(x) > length - 1)


def is_experiment_related(title):
    """Check if the title is experiment related."""
    name = title.split(":")[1].lower()

    long_keywords = [
        "method",
        "material",
        "experimental"
        # "materials and method",
        # "methods and material",
        # "experimental framework",
        # "experimental method",
        # "experimentation methodology",
        # "experimental design",
        # "experimental setup",
        # "experimental set-up",
        # "methods and experiment",
        # "experimental material",
        # "subjects and method",
        # "methods & materials",
        # "material and metod",
    ]

#     if section_name in short_keywords:
#         return True
    for k in long_keywords:
        if k in name:
            return True

    return False


def mentions_to_occurrence(raw_data,
                           term_column="entity",
                           factor_columns=None,
                           term_cleanup=None,
                           term_filter=None,
                           mention_filter=None,
                           aggregation_function=None,
                           dump_prefix=None):
    """Convert a raw mentions data into occurrence data.

    This function converts entity mentions into a dataframe
    indexed by unique entities. Each row contains aggregated data
    for different occurrence factors (sets of factor instances where
    the given term occurrs, e.g. sets of papers/section/paragraphs where the
    give term was mentioned).

    Parameters
    ----------
    raw_data : pandas.DataFrame
        Dataframe containing occurrence data with the following columns:
        one column for terms, one or more columns for occurrence factors (
        e.g. paper/section or paragraph of a term occurrence).
    term_column : str
        Name of the column with terms
    factor_columns : collection of str
        Set of column names containing occurrence factors (e.g.
        "paper"/"section"/"paragraph").
    term_cleanup : func, optional
        A clean-up function to be applied to every term
    term_filter : func, optional
        A filter function to apply to terms (e.g. include
        terms only with 2 or more symbols)
    mention_filter : func, optional
        A filter function to apply to occurrence factors (e.g. filter out
        all the occurrences in sections called "Methods")
    aggregation_function : func, optional
        Function to be applied to aggregated occurrence factors. By default,
        the constructor of `set`.
    dump_prefix : str, optional
        Prefix to use for dumping the resulting occurrence dataset.

    Returns
    -------
    occurence_data : pd.DataFrame
        Dataframe indexed by distinct terms containing aggregated occurrences
        of terms as columns (e.g. for each terms, sets of papers/sections/
        paragraphs) where the term occurs.
    factor_counts : dict
        Dictionary whose keys are factor column names (
        e.g. "paper"/"section"/"paragraph") and whose values are counts of
        unique factor instances (e.g. total number of papers/sections/
        paragraphs in the dataset)
    """
    print("Cleaning up the entities...")

    if factor_columns is None:
        factor_columns = []

    if term_cleanup is not None:
        raw_data[term_column] = raw_data[term_column].apply(term_cleanup)

    if term_filter is not None:
        raw_data = raw_data[term_filter(raw_data[term_column])]

    if mention_filter is not None:
        raw_data = raw_data[mention_filter(raw_data)]

    factor_counts = {}
    for factor_column in factor_columns:
        factor_counts[factor_column] = len(raw_data[factor_column].unique())

    print("Aggregating occurrences of entities....")
    if aggregation_function is None:
        aggregation_function = set

    occurence_data = raw_data.groupby(term_column).aggregate(
        lambda x: aggregation_function(x))

    if dump_prefix is not None:
        print("Saving the occurrence data....")
        with open("{}occurrence_data.pkl".format(dump_prefix), "wb") as f:
            pickle.dump(occurence_data, f)

        with open("{}counts.pkl".format(dump_prefix), "wb") as f:
            pickle.dump(factor_counts, f)

    return occurence_data, factor_counts


def aggregate_cord_entities(x, factors):
    """Aggregate a collection of entity mentions.

    Entity types are aggregated as lists (to preserve the multiplicity,
    e.g. how many times a given entity was recognized as a particular type).
    The rest of the input occurrence factors are aggregated as sets
    (e.g. sets of unique papers/sections/paragraphs).
    """
    result = {
        "entity_type": list(x.entity_type)
    }
    for f in factors:
        result[f] = set(x[f])

    return result


def prepare_occurrence_data(mentions_df=None,
                            mentions_path=None,
                            occurrence_data_path=None,
                            factor_counts_path=None):
    """Prepare mentions data for the co-occurrence analysis and dump.

    This function converts CORD-19 entity mentions into a dataframe
    indexed by unique entities. Each row contains aggregated data
    for different occurrence factors (sets of factor instances where
    the given term occurrs, e.g. sets of papers/section/paragraphs where the
    give term was mentioned).

    Parameters
    ----------
    mentions_df : pd.DataFrame, optional
        Dataframe containing occurrence data with the following columns:
        `entity`, `entity_type`, `occurrence` (occurrence in a paragraph
        identified with a string of format
        <paper_id>:<section_id>:<paragraph_id>). If not specified, the
        `mentions_path` argument will be used to load the mentions file.
    mentions_path : str, optional
        Path to a pickle file containing occurrence data of the shape
        described above.
    occurrence_data_path : str, optional
        Path to write the resulting aggregated occurrence data.
    factor_counts_path : str, optional
        Path to write the dictorary containing counts of different occurrence
        factors (papers, sections, paragraphs).

    Returns
    ------_
    occurence_data : pd.DataFrame
        Dataframe indexed by distinct terms containing aggregated occurrences
        of terms as columns (e.g. for each terms, sets of papers/sections/
        paragraphs) where the term occurs.
    counts : dict
        Dictionary whose keys are factor column names (
        i.e. "paper"/"section"/"paragraph") and whose values are counts of
        unique factor instances (e.g. total number of papers/sections/
        paragraphs in the dataset)
    """
    mentions = None
    if mentions_path:
        # Read raw mentions and transform them to the occurrence data
        if "pkl" in mentions_path:
            with open(mentions_path, "rb") as f:
                mentions = pickle.load(f)
        else:
            mentions = pd.read_csv(f)
    elif mentions_df is not None:
        mentions = mentions_df

    if mentions is not None:
        mentions = mentions[["entity", "entity_type", "occurrence"]]
        mentions["paper"] = mentions["occurrence"].apply(
            lambda x: x.split(":")[0])
        mentions["section"] = mentions["occurrence"].apply(
            lambda x: ":".join([x.split(":")[0], x.split(":")[1]]))
        mentions = mentions.rename(columns={"occurrence": "paragraph"})

        factors = ["paper", "section", "paragraph"]

        occurrence_data, counts = mentions_to_occurrence(
            mentions,
            term_column="entity",
            factor_columns=factors,
            term_cleanup=clean_up_entity,
            term_filter=lambda x: has_min_length(x, 3),
            aggregation_function=lambda x: aggregate_cord_entities(x, factors),
            mention_filter=lambda data: ~data["section"].apply(
                is_experiment_related))

        # Filter entities that occur only once (only in one paragraph,
        # usually represent noisy terms)
        occurrence_data = occurrence_data[occurrence_data["paragraph"].apply(
            lambda x: len(x) > 1)]
        if occurrence_data_path:
            print("Saving pre-calculated occurrence data....")
            with open(occurrence_data_path, "wb") as f:
                pickle.dump(occurrence_data, f)
        if factor_counts_path:
            with open(factor_counts_path, "wb") as f:
                pickle.dump(counts, f)

    elif occurrence_data_path is not None:
        with open(occurrence_data_path, "rb") as f:
            occurrence_data = pickle.load(f)
    else:
        occurrence_data = None
        counts = None

    if counts is None and factor_counts_path:
        with open(factor_counts_path, "rb") as f:
            counts = pickle.load(f)
    return occurrence_data, counts


def generate_curation_table(data):
    """Generate curation table from the raw co-occurrence data

    This function converts CORD-19 entity mentions into a dataframe
    indexed by unique entities. Each row contains aggregated data
    for different occurrence factors, i.e. papers/section/paragraphs,
    where the given term was mentioned).

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing occurrence data with the following columns:
        `entity`, `entity_type`, `occurrence` (occurrence in a paragraph
        identified with a string of format
        <paper_id>:<section_id>:<paragraph_id>).

    Returns
    -------
    result_data : pd.DataFrame
        Dataframe indexed by distinct terms containing aggregated occurrences
        of terms as columns (e.g. for each terms, sets of papers/sections/
        paragraphs) where the term occurs.
    counts : dict
        Dictionary whose keys are factor column names (
        i.e. "paper"/"section"/"paragraph") and whose values are counts of
        unique factor instances (e.g. total number of papers/sections/
        paragraphs in the dataset)
    """
    result_data, counts = prepare_occurrence_data(data)

    result_data = result_data.reset_index()
    result_data["paper_frequency"] = result_data[
        "paper"].transform(lambda x: len([str(p).split(":")[0] for p in x]))
    result_data["paper"] = result_data[
        "paper"].transform(lambda x: list(x))
    result_data["paragraph"] = result_data[
        "paragraph"].transform(lambda x: list(x))
    result_data["section"] = result_data[
        "section"].transform(lambda x: list(x))
    result_data["raw_entity_types"] = result_data[
        "entity_type"].transform(list)
    result_data["raw_frequency"] = result_data["entity_type"].apply(len)
    result_data["entity_type"] = result_data[
        "entity_type"].transform(
            lambda x: ", ".join(list(set(x))))
    return result_data, counts


def merge_with_ontology_linking(occurence_data,
                                factor_columns,
                                linking_df=None,
                                linking_path=None,
                                linked_occurrence_data_path=None):
    """Merge occurrence data with ontology linking data."""
    if factor_columns is None:
        factor_columns = ["paper", "section", "paragraph"]

    # Ontology linking
    linking = None
    if linking_path:
        # Open ontology linking files
        print("Loading the ontology linking...")
        if "pkl" in linking_path:
            with open(linking_path, "rb") as f:
                linking = pickle.load(f)
        else:
            linking = pd.read_csv(linking_path)
    elif linking_df is not None:
        linking = linking_df

    if linking is not None:
        linking = linking.rename(columns={"mention": "entity"})
        linking["concept"] = linking["concept"].apply(lambda x: x.lower())
        linking["entity"] = linking["entity"].apply(lambda x: x.lower())
        # The provided occcurence_data is expected to be lower cased and
        # the merge is performed on the 'entity' column and not the column one.

        print("Merging the occurrence data with the ontology linking...")

        # Merge occurrence data with the linking data
        occurence_data = occurence_data.reset_index()
        merged_data = occurence_data.merge(
            linking, on="entity", how="left")
        merged_data.loc[
            merged_data["concept"].isna(), "concept"] = merged_data[
                merged_data["concept"].isna()]["entity"]
        for col in factor_columns:
            merged_data[col] = merged_data[col].apply(lambda x: list(x))

        def aggregate_linking_data(x, factors):
            if x.name == "entity":
                return list(x)
            elif x.name in factors:
                return set(sum(x, []))
            elif x.name in ["uid", "definition", "taxonomy", "semantic_type"]:
                return list(x)[0]
            return sum(x, [])

        occurrence_data_linked = merged_data.groupby("concept").aggregate(
            lambda x: aggregate_linking_data(x, factor_columns))

        occurrence_data_linked = occurrence_data_linked.reset_index()
        occurrence_data_linked = occurrence_data_linked.rename(
            columns={
                "concept": "entity",
                "entity": "aggregated_entities"
            })
        occurrence_data_linked = occurrence_data_linked.set_index("entity")

        if linked_occurrence_data_path:
            with open(linked_occurrence_data_path, "wb") as f:
                print("Saving pre-calculated linked occurrence data....")
                pickle.dump(occurrence_data_linked, f)
        return occurrence_data_linked
    elif linked_occurrence_data_path:
        print("Loading linked occurrence data...")
        with open(linked_occurrence_data_path, "rb") as f:
            occurrence_data_linked = pickle.load(f)
    else:
        raise ValueError(
            "Neither linking data nor pre-computed linked occurrence "
            "data has been specified"
        )
    return occurrence_data_linked


def _configure_backends(backend_configs, graph):
    if backend_configs is None:
        backend_configs = dict()
    metrics_backend = (
        backend_configs["metrics"]
        if "metrics" in backend_configs else "networkx"
    )
    communities_backend = (
        backend_configs["communities"]
        if "communities" in backend_configs else "networkx"
    )
    paths_backend = (
        backend_configs["paths"]
        if "paths" in backend_configs else "networkx"
    )
    if metrics_backend == "neo4j":
        processor = BACKEND_MAPPING[metrics_backend]["metrics"](
            graph,
            backend_configs["driver"],
            backend_configs["node_label"],
            backend_configs["edge_label"],
            directed=False)
    else:
        processor = BACKEND_MAPPING[metrics_backend]["metrics"](
            graph, directed=False)

    if communities_backend == "neo4j":
        com_detector = BACKEND_MAPPING[communities_backend]["communities"](
            graph,
            backend_configs["driver"],
            backend_configs["node_label"],
            backend_configs["edge_label"],
            directed=False)
    else:
        com_detector = BACKEND_MAPPING[communities_backend]["communities"](
            graph, directed=False)

    if paths_backend == "neo4j":
        path_finder = BACKEND_MAPPING[paths_backend]["paths"](
            graph,
            backend_configs["driver"],
            backend_configs["node_label"],
            backend_configs["edge_label"],
            directed=False)
    else:
        path_finder = BACKEND_MAPPING[paths_backend]["paths"](
            graph, directed=False)
    pgframe_converter = BACKEND_MAPPING[paths_backend]["to_pgframe"]
    return processor, com_detector, path_finder, pgframe_converter


def generate_cooccurrence_analysis(occurrence_data, factor_counts,
                                   type_data=None, min_occurrences=1,
                                   n_most_frequent=None, keep=None,
                                   factors=None, cores=8,
                                   graph_dump_prefix=None,
                                   communities=True, remove_zero_mi=False,
                                   backend_configs=None,
                                   community_strategy="louvain"):
    """Generate co-occurrence analysis.

    This utility executes the entire pipeline of the co-occurrence analysis:
    it generates co-occurrence networks based on the input factors, yields
    various co-occurrence statistics (frequency, mutual-information-based
    scores) as edge attributes, computes various node centrality
    measures, node communities (and attaches them to the node attributes of
    the generated networks). Finally, it computes minimum spanning trees
    given the mutual-information-based distance scores (1 / NPMI). The function
    allows to dump the resulting graph objects using a pickle representation.

    Parameters
    ----------
    occurrence_data : pd.DataFrame
        Input occurrence data table. Rows represent unique entities (indexed
        by entity names), columns contain sets of aggregated occurrence factors
        (e.g. sets of papers/sections/paragraphs where the given term occurs).
    factor_counts : dict
        Dictionary whose keys are factor column names (
        i.e. "paper"/"section"/"paragraph") and whose values are counts of
        unique factor instances (e.g. total number of papers/sections/
        paragraphs in the dataset)
    type_data : pd.DataFrame, optional
        Table containing node types (these types are saved as node attributes)
    min_occurrences : int, optional
        Minimum co-occurrence frequency to consider (add as an edge to the co-
        occurrence network). By default every non-zero co-occurrence frequency
        yields an edge in the resulting network.
    n_most_frequent : int, optional
        Number of most frequent entitites to include in the co-occurrence
        network. By default is not set, therefore, all the terms from the
        occurrence table are included.
    keep : iterable
        Collection of entities to keep even if they are not included in N most
        frequent entities.
    factors : iterable, optional
        Set of factors to use for constructing co-occurrence networks
        (a network per factor is produced).
    cores : int, optional
        Number of cores to use during the parallel network generation.
    graph_dump_prefix : str
        Path prefix for dumping the generated networks (the edge
        list, edge attributes, node list and node attributes are saved).
    communities : bool, optional
        Flag indicating whether the community detection should be included
        in the analysis. By default True.
    remove_zero_mi : bool, optional
        Flag indicating whether edges with zero mutual-information scores
        (PPMI and NPMI) should be removed from the network (helps to sparsify
        the network, however may result in isolated nodes of high occurrence
        frequency).

    Returns
    -------
    graphs : dict of nx.DiGraph
        Dictionary whose keys are factor names and whose values are
        generated co-occurrence networks.
    trees : dict of nx.DiGraph
        Dictionary whose keys are factor names and whose values are
        minimum spanning trees of generated co-occurrence networks.
    """
    def compute_distance(x):
        return 1 / x if x > 0 else math.inf

    # Filter entities that occur only once (only in one paragraph, usually
    # represent noisy terms)
    if "paragraph" in factors:
        occurrence_data = occurrence_data[occurrence_data["paragraph"].apply(
            lambda x: len(x) >= min_occurrences)]
        occurrence_data["paragraph_frequency"] = occurrence_data[
            "paragraph"].apply(lambda x: len(x))
    if "section" in factors:
        occurrence_data["section_frequency"] = occurrence_data[
            "section"].apply(lambda x: len(x))
    if "paper" in factors:
        occurrence_data["paper_frequency"] = occurrence_data[
            "paper"].apply(lambda x: len(x))

    graphs = {}
    trees = {}
    for f in factors:
        print("-------------------------------")
        print("Factor: {}".format(f))
        print("-------------------------------")

        # Build a PGFrame from the occurrence data
        graph = PandasPGFrame()
        entity_nodes = occurrence_data.index
        graph.add_nodes(entity_nodes)
        graph.add_node_types({n: "Entity" for n in entity_nodes})
        graph.add_node_properties(occurrence_data[f], prop_type="category")
        graph.add_node_properties(
            occurrence_data["{}_frequency".format(f)], prop_type="numeric")

        # Select most frequent nodes
        nodes_to_include = None
        if n_most_frequent:
            nodes_to_include = graph._nodes.nlargest(
                n_most_frequent, "{}_frequency".format(f)).index
            graph = graph.subgraph(nodes=nodes_to_include)

        # Generate co-occurrence edges
        generator = CooccurrenceGenerator(graph)
        edges = generator.generate_from_nodes(
            f, total_factor_instances=factor_counts[f],
            compute_statistics=["frequency", "ppmi", "npmi"],
            parallelize=True, cores=cores)

        # Remove edges with zero mutual information
        if remove_zero_mi:
            edges = edges[edges["ppmi"] > 0]

        graph._edges = edges.drop(columns=["common_factors"])
        graph.edge_prop_as_numeric("frequency")
        graph.edge_prop_as_numeric("ppmi")
        graph.edge_prop_as_numeric("npmi")

        npmi_distance = edges["npmi"].apply(compute_distance)
        npmi_distance.name = "distance_npmi"
        graph.add_edge_properties(npmi_distance, "numeric")

        # Set entity types
        if type_data is not None:
            graph.add_node_properties(
                type_data.reset_index().rename(
                    columns={
                        "entity": "@id",
                        "type": "entity_type"
                    }).set_index("@id"),
                prop_type="category")

        # Set papers as props
        graph.remove_node_properties(f)
        if nodes_to_include is not None:
            paper_data = occurrence_data.loc[nodes_to_include, "paper"]
        else:
            paper_data = occurrence_data["paper"]
        graph.add_node_properties(
            paper_data.apply(lambda x: list(x)),
            prop_type="category")

        graphs[f] = graph

        processor, com_detector, path_finder, pgframe_converter =\
            _configure_backends(backend_configs, graph)

        # Compute centralities
        all_metrics = processor.compute_all_node_metrics(
            degree_weights=["frequency"],
            pagerank_weights=["frequency"])

        for metrics, data in all_metrics.items():
            for weight, values in data.items():
                prop = pd.DataFrame(
                    values.items(),
                    columns=["@id", "{}_{}".format(metrics, weight)])
                graph.add_node_properties(prop, prop_type="numeric")

        # Compute communitites
        frequency_partition = com_detector.detect_communities(
            strategy=community_strategy, weight="frequency")
        prop = pd.DataFrame(
            frequency_partition.items(),
            columns=["@id", "community_frequency"])
        graph.add_node_properties(prop, prop_type="numeric")

        npmi_partition = com_detector.detect_communities(
            strategy=community_strategy, weight="npmi")
        prop = pd.DataFrame(
            npmi_partition.items(), columns=["@id", "community_npmi"])
        graph.add_node_properties(prop, prop_type="numeric")

        # Compute minimum spanning tree
        tree = path_finder.minimum_spanning_tree(distance="distance_npmi")
        tree_pgframe = pgframe_converter(tree)
        trees[f] = tree_pgframe

        # Dump the generated PGFrame
        if graph_dump_prefix:
            graph.export_json("{}_{}_graph.json".format(graph_dump_prefix, f))
            tree_pgframe.export_json(
                "{}_{}_tree.json".format(graph_dump_prefix, f))
    return graphs, trees


def assign_raw_type(x):
    counts = {}
    for t in x:
        if t in counts:
            counts[t] += 1
        else:
            counts[t] = 1
    t = max(counts.items(), key=operator.itemgetter(1))[0]
    return t


def resolve_taxonomy_to_types(occurrence_data, mapping):
    """Assign entity types from hierarchies of NCIT classes.

    This function assigns a unique entity type to every entity
    using the ontology linking data (hierarchy, or taxonomy,
    of NCIT classes) according to the input type mapping. If
    a term was not linked, i.e. does not have such a taxonomy
    attached, raw entity types from the NER model are using (
    a unique entity type is chosen by the majority vote).

    Parameters
    ----------
    occurrence_data : pd.DataFrame
        Input occurrence data table. Rows represent unique entities (indexed
        by entity names), columns contain the following columns: `taxonomy`
        list containing a hierarchy of NCIT ontology classes of the given
        entity, `raw_entity_types` list of raw entity types provided by
        the NER model.
    mapping : dict
        Mapping whose keys are type names to be used, values are dictionaries
        with two keys: `include` and `exclude` specifying NCIT ontology
        classes to respectively include and exclude to/from when assigning
        the given type.

    Returns
    -------
    type_data : pd.DataFrame
        Dataframe indexed by unique entities, containing the column `type`
        specifying the assigned types.
    """
    def assign_mapped_type(x):
        taxonomy = (
            ast.literal_eval(x.taxonomy)
            if isinstance(x.taxonomy, str)
            else x.taxonomy
        )
        types = [el for _, el in taxonomy]
        result_type = None

        for target_type, taxonomy_classes in mapping.items():
            include_found = False
            exclude_found = False
            if "include" in taxonomy_classes:
                include_found = False
                for t in taxonomy_classes["include"]:
                    if t in types:
                        include_found = True
                        break
            if "exclude" in taxonomy_classes:
                for t in taxonomy_classes["exclude"]:
                    if t in types:
                        exclude_found = True
                        break
            if include_found and not exclude_found:
                result_type = target_type
                break

        if result_type is None:
            result_type = assign_raw_type(x.raw_entity_types)

        return result_type

    type_data = pd.DataFrame(index=occurrence_data.index, columns=["type"])

    known_taxonomy = occurrence_data[occurrence_data["taxonomy"].notna()]
    type_data.loc[known_taxonomy.index, "type"] = known_taxonomy.apply(
        assign_mapped_type, axis=1)

    unknown_taxonomy = occurrence_data[occurrence_data["taxonomy"].isna()]
    type_data.loc[unknown_taxonomy.index, "type"] = unknown_taxonomy[
        "raw_entity_types"].apply(assign_raw_type)
    return type_data


def link_ontology(linking, type_mapping, curated_table):
    """Merge the input occurrence table with the ontology linking.

    Parameters
    ----------
    linking : pd.DataFrame
        Datatable containing the linking data. The table includes
        the following columns: `mention` contains raw entities
        given by the NER model, `concept` linked ontology term,
        `uid` ID of the linked term in NCIT, `definition` definition
        of the linked term, `taxonomy` a list containing uid's and names
        of the parent ontology classes of the term.

    type_mapping : dict
        Mapping whose keys are type names to be used, values are dictionaries
        with two keys: `include` and `exclude` specifying NCIT ontology
        classes to respectively include and exclude to/from when assigning
        the given type.
    curated_table : pd.DataFrame
        Input occurrence data table. Rows represent unique entities (indexed
        by entity names), columns contain sets of aggregated occurrence factors
        (e.g. sets of papers/sections/paragraphs where the given term occurs),
        raw entity types (given by the NER model)

    Returns
    -------
    linked_table : pd.DataFrame
        The resulting table after grouping synonymical entities according to
        the ontology linking. The table is indexed by unqique linked entities
        containing the following columns: `paper`, `section`,
        `paragraph` representing aggregated factors where the term occurs,
        `aggregated_entities` set of raw entities linked to the given term,
        `uid` unique identifier in NCIT, if available, `definition`
        definition of the term in NCIT, `paper_frequency` number of unique
        papers where mentioned, `entity_type` a unique entity type per entity
        resolved using the ontology linking data (hierarchy, or taxonomy,
        of NCIT classes) according to the input type mapping.

    """
    linked_table = merge_with_ontology_linking(
        curated_table,
        factor_columns=["paper", "section", "paragraph"],
        linking_df=linking.rename(columns={"mention": "entity"}))

    linked_table = linked_table.reset_index()
    linked_table["paper_frequency"] = linked_table["paper"].apply(
        lambda x: len(x))
    linked_table["paper"] = linked_table["paper"].apply(lambda x: list(x))
    linked_table["section"] = linked_table["section"].apply(lambda x: list(x))
    linked_table["paragraph"] = linked_table["paragraph"].apply(
        lambda x: list(x))

    # Produce a dataframe with entity types according to type_mapping
    types = resolve_taxonomy_to_types(
        linked_table.set_index("entity"),
        type_mapping)
    linked_table = linked_table.merge(types, on="entity", how="left").rename(
        columns={"type": "entity_type"})
    linked_table["entity_type_label"] = linked_table["entity_type"]
    return linked_table


def generate_paper_lookup(graph):
    paper_table = {}
    paper_table = graph.get_node_property_values("paper")
    return paper_table


def build_cytoscape_data(graph_processor, positions=None):
    elements = []
    for node, properties in graph_processor.nodes(properties=True): 
        properties = properties.copy()
        data = {
            "id": node,
            "value": node,
            "name": node,
            "type": "node"
        }
        if 'paper' in properties:
            papers = properties["paper"]
            data["paper_frequency"] = len(papers)
            del properties["paper"]
        data.update(properties)

        element = {"data": data}
        if positions is not None:
            if node in positions:
                element["position"] = positions[node]

        elements.append(element)

    for s, t, properties in graph_processor.edges(properties=True):
        properties = properties.copy()
        s_name = s.replace(" ", "_")
        t_name = t.replace(" ", "_")
        data = {
            "id": f"{s_name}_{t_name}",
            "source": s,
            "target": t,
            "type": "edge"
        }
        data.update(properties)
        elements.append({"data": data})

    return elements


# def build_cytoscape_data(graph, positions=None):

#     elements = cytoscape_data(graph)

#     if positions is not None:
#         for el in elements["elements"]['nodes']:
#             if el["data"]["id"] in positions:
#                 el["position"] = positions[el["data"]["id"]]

#     elements = elements["elements"]['nodes'] + elements["elements"]['edges']
#     for element in elements:
#         element["data"]["id"] = (
#             str(element["data"]["source"] + '_' + element[
#                 "data"]["target"]).replace(" ", "_")
#             if "source" in element["data"]
#             else element["data"]["id"]
#         )
#         papers = []
#         if 'paper' in element["data"]:
#             papers = element["data"].pop("paper")
#             element["data"]["paper_frequency"] = len(papers)

#         if "source" in element["data"]:
#             element["data"]["type"] = "edge"
#         else:
#             element["data"]["type"] = "node"

#     return elements


def most_common(x):
    c = Counter(x)
    return c.most_common(1)[0][0]


CORD_ATTRS_RESOLVER = {
    "entity_type": most_common,
    "paper": lambda x: list(set(sum(x, []))),
    "degree_frequency": sum,
    "pagerank_frequency": max,
    "community_frequency": most_common,
    "community_npmi": most_common,
    "frequency": sum,
    "ppmi": max,
    "npmi": max,
    "distance_ppmi": min,
    "distance_npmi": min
}


def merge_attrs(source_attrs, collection_of_attrs, attr_resolver,
                attrs_to_ignore=None):
    """Merge two attribute dictionaries into the target using the input resolver.

    Parameters
    ----------
    source_attrs : dict
        Source dictionary with attributes (the other attributes will be
        merged into it and a new object will be returned)
    collection_of_attrs : iterable of dict
        Collection of dictionaries to merge into the target dictionary
    attr_resolver : dict
        Dictionary containing attribute resolvers, its keys are attribute
        names and its values are functions applied to the set of attribute
        values in order to resolve this set to a single value
    attrs_to_ignore : iterable, optional
        Set of attributes to ignore (will not be included in the merged
        node or edges incident to this merged node)
    """
    result = source_attrs.copy()

    if attrs_to_ignore is None:
        attrs_to_ignore = []

    all_keys = set(
        sum([list(attrs.keys()) for attrs in collection_of_attrs], []))

    def _preprocess(k, attrs):
        if k == "paper" and isinstance(attrs, str):
            return ast.literal_eval(attrs)
        else:
            return attrs

    for k in all_keys:
        if k not in attrs_to_ignore:
            if k in attr_resolver:
                result[k] = attr_resolver[k](
                    ([_preprocess(k, result[k])] if k in result else []) + [
                        _preprocess(k, attrs[k])
                        for attrs in collection_of_attrs if k in attrs
                    ]
                )
            else:
                result[k] = None
    return result


def merge_nodes(graph_processor, nodes_to_merge, new_name=None,
                attr_resolver=None):
    """Merge the input set of nodes.

    Parameters
    ----------
    graph_processor : GraphProcessor
        Input graph object
    nodes_to_merge: iterable
        Collection of node IDs to merge
    new_name : str, optional
        New name to use for the result of merging
    attr_resolver : dict, optional
        Dictionary containing attribute resolvers, its keys are attribute
        names and its values are functions applied to the set of attribute
        values in order to resolve this set to a single value

    Returns
    -------
    graph : nx.Graph
        Resulting graph (references to the input graph, if `copy` is False,
        or to another object if `copy` is True).
    """
    if len(nodes_to_merge) < 2:
        raise ValueError("At least two nodes are required for merging")

    if attr_resolver is None:
        raise ValueError("Attribute resolver should be provided")

    # We merge everything into the target node
    if new_name is None:
        new_name = nodes_to_merge[0]

    if new_name not in graph_processor.nodes():
        graph_processor.rename_nodes(
            {nodes_to_merge[0]: new_name})
        nodes_to_merge = nodes_to_merge[1:]

    target_node = new_name
    other_nodes = [n for n in nodes_to_merge if n != target_node]

    # Resolve node attrs
    graph_processor.set_node_properties(
        target_node,
        merge_attrs(
            graph_processor.get_node(target_node),
            [graph_processor.get_node(n) for n in other_nodes],
            attr_resolver)
    )

    # Merge edges
    edge_attrs = {}

    for n in other_nodes:
        neighbors = graph_processor.neighbors(n)
        for neighbor in neighbors:
            if neighbor != target_node and neighbor not in other_nodes:
                if neighbor in edge_attrs:
                    edge_attrs[neighbor].append(
                        graph_processor.get_edge(n, neighbor))
                else:
                    edge_attrs[neighbor] = [
                        graph_processor.get_edge(n, neighbor)
                    ]

    edges = graph_processor.edges()
    for k, v in edge_attrs.items():
        target_neighbors = graph_processor.neighbors(target_node)
        if k not in target_neighbors:
            graph_processor.add_edge(
                target_node, k,
                merge_attrs({}, v, attr_resolver))
        else:
            graph_processor.set_edge_properties(
                target_node, k,
                merge_attrs(
                    graph_processor.get_edge(target_node, k),
                    v, attr_resolver))

    for n in other_nodes:
        graph_processor.remove_node(n)

    return graph_processor.graph


def download_from_nexus(uri, config_file_path, output_path, nexus_endpoint, nexus_bucket, unzip=False):
    forge = KnowledgeGraphForge(config_file_path, endpoint=nexus_endpoint, bucket=nexus_bucket)
    dataset = forge.retrieve(id=uri)
    print(f"Downloading the file to {output_path}/{dataset.distribution.name}")
    forge.download(dataset, path=output_path, overwrite=True, follow="distribution.contentUrl")
    if unzip:
        print(f"Decompressing ...")
        with zipfile.ZipFile(f"{output_path}/{dataset.distribution.name}", 'r') as zip_ref:
            zip_ref.extractall(output_path)
    return dataset
