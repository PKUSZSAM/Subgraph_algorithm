import networkx as nx
import argparse
from glob import glob

def match_func(G1, G2):
    return nx.is_isomorphic(G1, G2)

def subgraph_matching(G1_list, G2_list):
    # do a bijection matching
    if len(G1_list) != len(G2_list):
        return False
    
    remaining = G2_list[:]

    for item1 in G1_list:
        matched = False
        for item2 in remaining:
            if match_func(item1, item2):
                remaining.remove(item2)
                matched = True
                break
        if not matched:
            return False
        
    return not remaining

argparser = argparse.ArgumentParser()
argparser.add_argument('--struct1', type=str, required=True, help='Path to the first structure file')
argparser.add_argument('--struct2', type=str, required=True, help='Path to the second structure file')
args = argparser.parse_args()

path_suffix = '/graph_types/*/*.gml'
G1_path = glob(args.struct1+path_suffix)
G2_path = glob(args.struct2+path_suffix)

G1_list = [nx.read_graphml(path) for path in G1_path]
G2_list = [nx.read_graphml(path) for path in G2_path]

print(subgraph_matching(G1_list, G2_list))
