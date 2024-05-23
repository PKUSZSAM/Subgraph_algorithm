import networkx as nx
import random
import argparse

def regist_attr(G):
    center = nx.center(G)[0]
    G.nodes[center]['graph_distance'] = 0

    for node in G.nodes:
        if node != center:
            G.nodes[node]['graph_distance'] = nx.shortest_path_length(G, center, node)

    # regist degree
    for node in G.nodes:
        G.nodes[node]['degree'] = G.degree[node]
    
    return

def gdist_isomorphism(G1, G2):
    """
    Compute the graph edit distance between two graphs layer by layer, 
    return the longest graph distance value of isomorphism subgraph.  
    """
    # compare the isomorphism layer by layer
    layer = 0

    for index in range(1, len(G1.nodes)):
        if nx.is_isomorphic(G1.subgraph([node for node in G1.nodes if G1.nodes[node]['graph_distance'] <= index]), G2.subgraph([node for node in G2.nodes if G2.nodes[node]['graph_distance'] <= index])):
            layer = index
        else:
            break
    
    return layer

def subgraph_coarsen(G, layer):
    """
    Coarse the graph based on the isomorphism subgraph.
    """
    merged_nodes = generate_unique_node_id(G, len([node for node in G.nodes if G.nodes[node]['graph_distance'] == layer - 1]))
    merged_nodes_mapping={}
    tmp_node_list = [node for node in G.nodes if G.nodes[node]['graph_distance'] == layer - 1]
    for index, node in enumerate(merged_nodes):
        merged_nodes_mapping[node] = [neighbor for neighbor in G.neighbors(tmp_node_list[index]) if G.nodes[neighbor]['graph_distance'] == layer]

    rm_nodes = [node for node in G.nodes if G.nodes[node]['graph_distance'] <= layer]
    link_nodes = [node for node in G.nodes if G.nodes[node]['graph_distance'] == layer]
    # get the dist of node and its neighbors with graph_distance = layer + 1
    links = {}
    for node in link_nodes:
        links[node] = [neighbor for neighbor in G.neighbors(node) if G.nodes[neighbor]['graph_distance'] == layer + 1]
    
    G.add_nodes_from(merged_nodes, graph_distance = layer)

    for merged_node in merged_nodes:
        for neighbor in merged_nodes_mapping[merged_node]:
            for links_node in links[neighbor]:
                G.add_edge(merged_node, links_node)

    G.remove_nodes_from(rm_nodes)

    for node in merged_nodes:
        G.nodes[node]['degree'] = G.degree[node]
        G.nodes[node]['neighbors_degree'] = [G.nodes[neighbor]['degree'] for neighbor in G.neighbors(node)]

    return G

def generate_unique_node_id(G, num):
    """
    Generate several unique node id for the merged node.
    """
    unique_id_list = []
    # check whether the node id is unique
    for i in range(num):
        while True:
            unique_id = str(random.randint(1,1000) + i + 1)
            if unique_id not in G.nodes:
                unique_id_list.append(unique_id)
                break
    return unique_id_list

def GED_verification(_G1, _G2, GED, layer, loop_max=10000):
    """
    Verify the GED between two subgraphs.
    Now only consider that the difference between two subgraphs are only on the last layer.
    """

    if any((_G1.nodes[node]['graph_distance'] - layer) > 1 for node in _G1.nodes):
        NotImplementedError('The difference between two subgraphs now should only on the last layer.')

    node_diff = abs(len(_G1.nodes) - len(_G2.nodes))
    edge_diff = abs(len(_G1.edges) - len(_G2.edges))

    if node_diff + edge_diff > GED:
        return False
    else:
        # do the loop for the node and edge addition and substitution

        loop = 0

        while loop < loop_max:
            residue_GED = GED
            G1 = _G1.copy()
            G2 = _G2.copy()
        # add the node on the last layer
            if node_diff > 0:
                if len(G1.nodes) > len(G2.nodes):
                    for _ in range(node_diff):
                        new_node_id = generate_unique_node_id(G2,1)[0]
                        G2.add_node(new_node_id, graph_distance = layer+1)
                        # node should has at least one edge, add one edge from the new node to the layer node.
                        G2.add_edge(new_node_id, random.choice([node for node in G2.nodes if G2.nodes[node]['graph_distance'] == layer]))
                elif len(G1.nodes) < len(G2.nodes):
                    for _ in range(node_diff):
                        new_node_id = generate_unique_node_id(G1,1)[0]
                        G1.add_node(new_node_id, graph_distance = layer+1)
                        G1.add_edge(new_node_id, random.choice([node for node in G1.nodes if G1.nodes[node]['graph_distance'] == layer]))
                residue_GED = residue_GED - node_diff*2
                if residue_GED == 0:
                    if nx.is_isomorphic(G1, G2):
                        return True
                    else:
                        loop += 1
                        continue
                elif residue_GED < 0:
                    loop += 1
                    continue
                
            # now the node_diff should be 0, node will not be modified, only add or substitute the edge.
            # edge should only connect the node between layer and layer+1

            edge_diff = abs(len(G1.edges) - len(G2.edges))
            if edge_diff > 0:
                if len(G1.edges) > len(G2.edges):
                    for _ in range(edge_diff):
                        G2.add_edge(random.choice([node for node in G2.nodes if G2.nodes[node]['graph_distance'] == layer]), random.choice([node for node in G2.nodes if G2.nodes[node]['graph_distance'] == layer+1]))
                elif len(G1.edges) < len(G2.edges):
                    for _ in range(edge_diff):
                        G1.add_edge(random.choice([node for node in G2.nodes if G2.nodes[node]['graph_distance'] == layer]), random.choice([node for node in G2.nodes if G2.nodes[node]['graph_distance'] == layer+1]))
                residue_GED = residue_GED - edge_diff
                if residue_GED == 0:
                    if nx.is_isomorphic(G1, G2):
                        return True
                    else:
                        loop += 1
                        continue
                elif residue_GED < 0:
                    loop += 1
                    continue
                    
            # now do the edge substitution
            # here the edge substitution include two edit operators
            if residue_GED % 2 != 0:
                return False
            else:
                residue_GED = residue_GED // 2
            for _ in range(residue_GED):
                node1 = random.choice([node for node in G1.nodes if G1.nodes[node]['graph_distance'] == layer])
                node2 = random.choice([node for node in G1.nodes if G1.nodes[node]['graph_distance'] == layer+1])
                G1.remove_edge(node1, random.choice([neighbor for neighbor in G1.neighbors(node1)]))
                G1.add_edge(node1, node2)
            if nx.is_isomorphic(G1, G2):
                return True

            loop += 1

    return False
        


argparser = argparse.ArgumentParser()
argparser.add_argument('--subgraph1', type=str, required=True, help='Path to the first subgraph gml file')
argparser.add_argument('--subgraph2', type=str, required=True, help='Path to the second subgraph gml file')
argparser.add_argument('--seed', type=int, default=10, help='Timeout for the GED computation')
argparser.add_argument('--upper_bound', type=int, default=4, help='Upper bound for the GED')
args = argparser.parse_args()

random.seed(args.seed)

G1 = nx.read_graphml(args.subgraph1)
G2 = nx.read_graphml(args.subgraph2)

regist_attr(G1)
regist_attr(G2)
layer = gdist_isomorphism(G1, G2)

match_flag = False
for ged in range(1, args.upper_bound+1):
    if GED_verification(G1, G2, ged, layer):
        print('The GED between two subgraphs is %d' % ged)
        # finish the verification
        match_flag = True
        break

if not match_flag:
    print('The GED between two subgraphs is larger than %d' % ged)
