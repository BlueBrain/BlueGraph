from bluegraph.core.analyse.communities import CommunityDetector

from graph_tool.inference import (minimize_blockmodel_dl,
                                  minimize_nested_blockmodel_dl)

from ..io import GTGraphProcessor


class GTCommunityDetector(GTGraphProcessor, CommunityDetector):
    """Graph-tool-based community detection interface.

    https://graph-tool.skewed.de/static/doc/demos/inference/inference.html
    """
    def _run_louvain(self, **kwargs):
        raise CommunityDetector.PartitionError(
            "Louvain algorithm is not implemented "
            "for graph-tool-based graphs")

    def _run_girvan_newman(self, weight=None, n_communitites=2,
                           intermediate=False):
        raise CommunityDetector.PartitionError(
            "Girvan-Newman algorithm is not implemented "
            "for Neo4j-based graphs")

    def _run_stochastic_block_model(self, weight=None,
                                    weight_model="discrete-binomial",
                                    nested=False,
                                    min_communities=None, max_communities=None,
                                    **kwargs):
        state_args = dict()
        if weight:
            state_args = dict(
                recs=[self.graph.ep[weight]],
                rec_types=[weight_model])

        if not nested and not weight:
            state = minimize_blockmodel_dl(
                self.graph, B_min=min_communities, B_max=max_communities)
            partition = state.get_blocks()
        else:
            state = minimize_nested_blockmodel_dl(
                self.graph, B_min=min_communities, B_max=max_communities,
                state_args=state_args)
            blocks = [l.get_blocks().a for l in state.get_levels()]
            partition = []
            for el in blocks[0]:
                communities = [el]
                prev_block = el
                if len(blocks) > 1:
                    for b in blocks[1:]:
                        communities.append(b[prev_block])
                        prev_block = b[prev_block]
                else:
                    communities = [el]
                communities.reverse()
                if not nested:
                    partition.append(communities[-1])
                else:
                    partition.append(communities)
        return partition

    def _run_label_propagation(self, **kwargs):
        raise CommunityDetector.PartitionError(
            "Label propagation algorithm is not implemented "
            "for Neo4j-based graphs")

    def _compute_modularity(self, partition, weight=None):
        pass

    def _compute_performance(self, partition, weight=None):
        pass

    def _compute_coverage(self, partition, weight=None):
        pass
