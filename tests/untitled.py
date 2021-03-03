MATCH (n1:TestNode), (n2:TestNode)
WHERE id(n1) <= id(n2)
WITH n1, n2
OPTIONAL MATCH (n1)-[r:TestEdge]-(n2)
WITH collect([n1, n2, r]) as data, collect(DISTINCT n1) as nodes
WITH [el IN data WHERE el[0].louvain_community <> el[1].louvain_community AND el[2] IS NULL] as non_interedges, [el in data WHERE el[0].louvain_community = el[1].louvain_community AND NOT el[2] IS NULL| el] as intraedges, nodes
RETURN toFloat(size(intraedges) + size(non_interedges)) / (toFloat(size(nodes) * (size(nodes) - 1)) / 2)