"""Small module for data preparation and network generation for COVID-19 paper."""
import operator
import pickle

import pandas as pd
import networkx as nx
import numpy as np

from collections import Counter

from networkx.readwrite.json_graph.cytoscape import cytoscape_data

from kganalytics.network_generation import generate_comention_network
from kganalytics.data_preparation import (mentions_to_occurrence,
                                        is_experiment_related,
                                        dummy_clean_up,
                                        is_not_single_letter)
from kganalytics.metrics import (compute_degree_centrality,
                               compute_pagerank_centrality,
                               compute_betweenness_centrality,
                               detect_communities,
                               compute_all_metrics)

from kganalytics.export import (save_nodes,
                                save_to_gephi,
                                save_network)
from kganalytics.paths import (minimum_spanning_tree,
                             top_n_paths,
                             top_n_tripaths,
                             single_shortest_path)


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

def dummy_clean_up(s):
    """Dummy clean-up to remove some errors from NER."""
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


def is_not_single_letter(entities):
    """Check if a term is not a single letter."""
    return entities.apply(lambda x: len(x) > 1)


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


def subgraph_by_types(graph, type_data, types_to_include, types_to_exclude=None, include_nodes=None):
    if include_nodes is None:
        include_nodes = []
    
    if types_to_exclude is None:
        types_to_exclude = []

    def has_needed_type(n):
#         print(n)
        n_types = type_data[n]
#         print(n_types)
        include_found = False
        exclude_found = False
        for t in n_types:
            if t in types_to_include:
                include_found = True
            if t in types_to_exclude:
                exclude_found = True
        return include_found and not exclude_found
    
    nodes = [
        n
        for n in graph.nodes()
        if has_needed_type(n) or n in include_nodes
    ]
    return nx.Graph(graph.subgraph(nodes))


def graph_from_paths(paths, source_graph=None):
    nodes = set()
    edges = set()
    for p in paths:
        for i in range(1, len(p)):
            nodes.add(p[i - 1])
            edges.add((p[i - 1], p[i]))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    if source_graph is not None:
        # Graph are asumed to be hemogeneous
        attrs = source_graph.nodes[list(nodes)[0]].keys()
        for k in attrs:
            nx.set_node_attributes(
                graph, {n: source_graph.nodes[n][k] for n in nodes}, k)
        edge_attrs = source_graph.edges[list(edges)[0]].keys()
        for k in edge_attrs:
            nx.set_edge_attributes(
                graph, {e: source_graph.edges[e][k] for e in edges}, k)
    return graph


def label_paths(graph, paths, label):
    path_group = dict()
    
    labeled_nodes = list()
    
    nodes = set()
    edges = set()
    for p in paths:
        nodes.add(p[0])
        for i in range(1, len(p)):
            nodes.add(p[i])
            edges.add((p[i - 1], p[i]))
    
    for n in graph.nodes():
        if n in nodes:
            path_group[n] = 1
            labeled_nodes.append(n)
        else:
            path_group[n] = 0
    nx.set_node_attributes(graph, path_group, label)

    path_group = dict()
    for e in graph.edges():
        if e in edges:
            path_group[e] = 1
        else:
            path_group[e] = 0
    nx.set_edge_attributes(graph, path_group, label)
    return labeled_nodes


def aggregate_cord_entities(x, factors):
    result = {
        "entity_type": list(x.entity_type)
    }
    for f in factors:
        result[f] = set(x[f])
    
    return result


def prepare_occurrence_data(mentions_df=None,
                 mentions_path=None,
                 occurrence_data_path=None,
                 factors=None,
                 factor_counts_path=None):
    """Prepare mentions data for the co-occurrence analysis."""    
    if factors is None:
        # use all factors
        factors = ["paper", "section", "paragraph"]

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
        mentions = mentions[["entity", "entity_type", "paper_id"]]
        mentions["paper"] = mentions["paper_id"].apply(
        lambda x: x.split(":")[0])
        mentions["section"] = mentions["paper_id"].apply(
            lambda x: ":".join([x.split(":")[0], x.split(":")[1]]))
        mentions = mentions.rename(columns={"paper_id": "paragraph"})
        occurrence_data, counts = mentions_to_occurrence(
            mentions,
            term_column="entity",
            factor_columns=factors,
            term_cleanup=dummy_clean_up,
            term_filter=is_not_single_letter,
            aggregation_function=lambda x: aggregate_cord_entities(x, factors),
            mention_filter=lambda data: ~data["section"].apply(is_experiment_related))

        # Filter entities that occur only once (only in one paragraph, usually represent noisy terms)
        occurrence_data = occurrence_data[occurrence_data["paragraph"].apply(lambda x: len(x) > 1)]
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
#         raise ValueError("Neither mention nor occurrence data have been provided")
        occurrence_data = None
        counts = None
    
    if counts is None and factor_counts_path:
         with open(factor_counts_path, "rb") as f:
            counts = pickle.load(f)
    return occurrence_data, counts


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
        linking["entity"] = linking["entity"].apply(lambda x: x.lower()) # The provided occcurence_data is expected to be lower cased and the merge is performed on the 'entity' column and not the column one.
        
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
            elif x.name in ["uid", "definition", "_subclassof_label", "_type_label"]:
                return list(x)[0]
            return sum(x, [])

        occurrence_data_linked = merged_data.groupby("concept").aggregate(
            lambda x: aggregate_linking_data(x, factor_columns))

        occurrence_data_linked = occurrence_data_linked.reset_index()
        occurrence_data_linked = occurrence_data_linked.rename(
            columns={
                "concept": "entity",
                "entity": "aggregated_entities",
                "entity_type": "raw_entity_types",
                "_type_label": "semantic_type",
                "_subclassof_label": "taxonomy"
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
            "Neither linking data nor pre-computed linked occurrence data has been specified")
    return occurrence_data_linked
    
def generate_comention_analysis(occurrence_data, counts, type_data=None, min_occurrences=1,
                                 n_most_frequent=None, factors=None, cores=None, graph_dump_prefix=None):
    # Filter entities that occur only once (only in one paragraph, usually represent noisy terms)
    occurrence_data = occurrence_data[occurrence_data["paragraph"].apply(lambda x: len(x) >= min_occurrences)]
    occurrence_data["paragraph_frequency"] = occurrence_data["paragraph"].apply(lambda x: len(x))
    occurrence_data["section_frequency"] = occurrence_data["section"].apply(lambda x: len(x))
    occurrence_data["paper_frequency"] = occurrence_data["paper"].apply(lambda x: len(x))

    graphs = {}
    trees = {}
    for f in factors:
        print("-------------------------------")
        print("Factor: {}".format(f))
        print("-------------------------------")
        
        if cores is None:
            cores = 8
        graphs[f] = generate_comention_network(
            occurrence_data,
            f,
            counts[f],
            n_most_frequent=n_most_frequent,
            dump_path=(
                "{}_{}_edge_list.pkl".format(graph_dump_prefix, f)
                if graph_dump_prefix else None
            ),
            parallelize=True,
            cores=cores)
        
        # Remove edges with zero mutual information
        edges_to_remove = [
            e
            for e in graphs[f].edges()
            if graphs[f].edges[e]["ppmi"] == 0
        ]
        for s, t in edges_to_remove:
            graphs[f].remove_edge(s, t)

        # Set entity types
        if type_data is not None:
            type_dict = type_data.to_dict()["type"]
            nx.set_node_attributes(
                graphs[f],
                type_dict,
                "entity_type")
            
        # Set factors as attrs
        nx.set_node_attributes(
            graphs[f],
            occurrence_data["paper"].apply(lambda x: list(x)).to_dict(),
            "paper")
            
        compute_all_metrics(
            graphs[f],
            degree_weights=["frequency"],
            betweenness_weights=[],
            community_weights=["frequency", "npmi"],
            print_summary=True)
    
        # Compute a spanning tree
        print("Computing the minimum spanning tree...")
        trees[f] = minimum_spanning_tree(graphs[f], weight="distance_npmi")
        nx.set_node_attributes(
            trees[f],
            type_dict,
            "entity_type")
        
        if graph_dump_prefix:
            save_network(trees[f], "{}_{}_tree".format(graph_dump_prefix, f))
        
        if graph_dump_prefix:
            save_nodes(
                graphs[f],
                "{}_{}_node_list.pkl".format(graph_dump_prefix, f))

    return graphs, trees
    

def generate_full_analysis(mentions_df=None, mentions_path=None,
                           occurrence_data_path=None, factor_counts_path=None,
                           linking_df=None, linking_path=None,
                           linked_occurrence_data_path=None,
                           type_mapping=None, type_data_path=None,
                           n_most_frequent=None, min_occurrences=None,
                           graph_dump_prefix=None, gephi_dump_prefix=None,
                           factors=None, cores=None):
    
    if factors is None:
        factors = ["paper", "section", "paragraph"]

    # Converting mention data into occurrences
    occurrence_data, counts = prepare_occurrence_data(
        mentions_df=mentions_df,
        mentions_path=mentions_path,
        occurrence_data_path=occurrence_data_path,
        factors=factors,
        factor_counts_path=factor_counts_path)

#     # Ontology linking
    occurrence_data_linked = merge_with_ontology_linking(
        occurrence_data,
        factor_columns=factors,
        linking_df=linking_df,
        linking_path=linking_path,
        linked_occurrence_data_path=linked_occurrence_data_path)

    
    # Peform type mapping according to the provided dictionary
    if type_data_path:
        type_data = pd.read_pickle(type_data_path)
    elif type_mapping is not None:
        type_data = resolve_taxonomy_to_types(
            occurrence_data_linked, type_mapping)
        if type_data_path:
            type_data.to_pickle(type_data_path)
  
    graphs, trees = generate_comention_analysis(
        occurrence_data_linked, counts,
        min_occurrences=min_occurrences,
        type_data=type_data,
        n_most_frequent=n_most_frequent,
        graph_dump_prefix=graph_dump_prefix,
        factors=factors,
        cores=cores)
    
    for f in factors:
        if gephi_dump_prefix:
            node_attr_mapping = {
                "degree_frequency": "Degree",
                "community_npmi": "Community",
            }
            if type_data is not None:
                node_attr_mapping["entity_type"] = "Type"

            edge_attr_mapping={
                "npmi": "Weight"
            }

            save_to_gephi(
                graphs[f],
                "{}_graph_{}".format(gephi_dump_prefix, f), 
                node_attr_mapping=node_attr_mapping,
                edge_attr_mapping=edge_attr_mapping)

            save_to_gephi(
                trees[f],
                "{}_tree_{}".format(gephi_dump_prefix, f), 
                node_attr_mapping=node_attr_mapping,
                edge_attr_mapping=edge_attr_mapping)

    return graphs, trees


def resolve_taxonomy_to_types(occurrence_data, mapping):

    def assign_raw_type(x):
        counts = {}
        for t in x:
            if t in counts:
                counts[t] +=1
            else:
                counts[t] = 1
        return max(counts.items(), key=operator.itemgetter(1))[0]
    
    def assign_mapped_type(x):
        types = [el for _, el in x.taxonomy]
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
    type_data.loc[known_taxonomy.index, "type"] = known_taxonomy.apply(assign_mapped_type, axis=1)
    
    unknown_taxonomy = occurrence_data[occurrence_data["taxonomy"].isna()]
    type_data.loc[unknown_taxonomy.index, "type"] = unknown_taxonomy["raw_entity_types"].apply(assign_raw_type)
    return type_data


def generate_curation_table(filtered_table_extractions):
    filtered_table_extractions, counts = prepare_occurrence_data(filtered_table_extractions)
    filtered_table_extractions = filtered_table_extractions.reset_index()
    filtered_table_extractions["paper_frequency"] = filtered_table_extractions["paper"].transform(lambda x:  len([str(p).split(":")[0] for p in x]))
    filtered_table_extractions["paper"] = filtered_table_extractions["paper"].transform(lambda x:  list(x))
    filtered_table_extractions["paragraph"] = filtered_table_extractions["paragraph"].transform(lambda x:  list(x))
    filtered_table_extractions["section"] = filtered_table_extractions["section"].transform(lambda x:  list(x))
    filtered_table_extractions = filtered_table_extractions.rename(columns={"entity_type": "raw_entity_types"})
    filtered_table_extractions["entity_type"] = filtered_table_extractions["raw_entity_types"].transform(
        lambda x:  ", ".join(list(set(x))))
    return filtered_table_extractions, counts


def link_ontology(linking, type_mapping, curated_table):
    curated_table = curated_table.drop(columns=["entity_type"])
    curated_table = curated_table.rename(columns={"raw_entity_types": "entity_type"})
    linked_table = merge_with_ontology_linking(
        curated_table,
        factor_columns=["paper", "section", "paragraph"],
        linking_df=linking.rename(columns={"mention": "entity"}))

    linked_table = linked_table.reset_index()
    linked_table["paper_frequency"] = linked_table["paper"].apply(lambda x: len(x))
    linked_table["paper"] = linked_table["paper"].apply(lambda x: list(x))
    linked_table["section"] = linked_table["section"].apply(lambda x: list(x))
    linked_table["paragraph"] = linked_table["paragraph"].apply(lambda x: list(x))
    
    # Produce a dataframe with entity types according to type_mapping
    types = resolve_taxonomy_to_types(
        linked_table.rename(
            columns={"entity_type": "raw_entity_types"}).set_index("entity"), type_mapping)
    linked_table = linked_table.merge(types, on="entity", how="left").rename(columns={"type": "entity_type"})
    linked_table["entity_type_label"] = linked_table["entity_type"] 
    return linked_table


def generate_paper_lookup(graph):
    paper_table = {}
    for n in graph.nodes():
        if "paper" in graph.nodes[n]:
            paper_table[n] = list(graph.nodes[n]["paper"])
    return paper_table


def build_cytoscape_data(graph, positions=None):
    elements = cytoscape_data(graph)
    
    if positions is not None:
        for el in elements["elements"]['nodes']:
            if el["data"]["id"] in positions:
                el["position"] = positions[el["data"]["id"]]
    
    elements = elements["elements"]['nodes'] + elements["elements"]['edges']
    for element in elements:
        element["data"]["id"] = (
            str(element["data"]["source"] + '_' + element["data"]["target"]).replace(" ","_")
            if "source" in element["data"]
            else element["data"]["id"]
        )
        papers = []
        if 'paper' in element["data"]:
            papers = element["data"].pop("paper")
            element["data"]["paper_frequency"] = len(papers)

        if "source" in element["data"]:
            element["data"]["type"] = "edge"
        else:
            element["data"]["type"] = "node"

    return elements


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
