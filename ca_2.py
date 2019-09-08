import networkx as nx
import numpy as np
import pandas as pd
import pylab

def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if not start in graph.keys():
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths

Df = pd.read_csv("table.csv")
graph_edges={x:[] for x in Df["Activity"]}
weights ={k:int(v) for k,v in zip(Df["Activity"],Df["Duration"])} #Duration

for index, row in Df.iterrows():
    if str(row["Predecessors"])=='nan':
       pass
    else:
        acts=str(row["Predecessors"]).split(" ")
        for act in acts:
            graph_edges[act].append(row['Activity'])

# Critical Path calculation
paths = find_all_paths(graph_edges, 'St', 'End')
time_t=[sum([weights[z] for z in path]) for path in paths]
index_p = time_t.index(max(time_t))

# ------------------------------------------------------------------------------------------------------------------------------
#65~70 lines of Code added
#Calculation of LS,ES,LF,EF

node_data = {k:{"ES":0,"EF":0,"LS":0,"LF":0} for k in graph_edges.keys()} # Keeps the ES EF LS LF values
pred={k:str(v).split(" ") for k,v in zip(Df["Activity"],Df["Predecessors"])}
stack=[]
visited={k:False for  k in graph_edges.keys()}


# What is Topological Sort? --> https://www.geeksforgeeks.org/topological-sorting/
# Topological Sort is not exactly required as we have taken "St" and "End" node which automatically handles the no predcessor requirement
# order and hence DFS can also be used
# Customized for personal use (Only stack is list, rest are dictionaries).
def topologicalSortUtil(graph_edges,v,visited,stack):
    visited[v] = True
    for i in graph_edges[v]:
        if visited[i] == False:
            topologicalSortUtil(graph_edges,i,visited,stack)
    stack.insert(0,v)

def topologicalSort(graph_edges,visited,stack):
    for i in graph_edges.keys():
        if visited[i] == False:
            topologicalSortUtil(graph_edges,i,visited,stack)

topologicalSort(graph_edges,visited,stack)
print(stack)#Stack successful
stack.pop(0) #Since using stack as vector, if 0 is unwanted reverse the stack list.

while(stack!=[]):
    top = stack[0];
    max_f = np.NINF
    for i in pred[top]:
        if(max_f < node_data[i]["EF"]):
            max_f = node_data[i]["EF"]

    node_data[top]["ES"] = max_f
    node_data[top]["EF"] = max_f + weights[top]
    stack.pop(0)

# print(node_data) ES & EF Successful

visited={k:False for  k in graph_edges.keys()}
topologicalSort(graph_edges,visited,stack)
r_stack=stack[::-1]
node_data[r_stack[0]]["LS"]=node_data[r_stack[0]]["ES"]
node_data[r_stack[0]]["LF"]=node_data[r_stack[0]]["EF"]
r_stack.pop(0)

while(r_stack!=[]):
    top = r_stack[0];
    min_s = np.Inf
    for i in graph_edges[top]:
        if(min_s > node_data[i]["LS"]):
            min_s = node_data[i]["LS"]

    node_data[top]["LF"] = min_s
    node_data[top]["LS"] = min_s - weights[top]
    r_stack.pop(0)

# print(node_data) ES & EF Successful
#Printing Table
print("Node\tES\tEF\tLS\tLF")
for  k,v in node_data.items():
    print(f'{k}\t{v["ES"]}\t{v["EF"]}\t{v["LS"]}\t{v["LF"]}')

#Plotting
G = nx.DiGraph(directed=True)
G.add_nodes_from(graph_edges.keys())
for key in graph_edges.keys():
    G.add_edges_from([(key,v) for v in graph_edges[key]], weight=weights[key])

red_edges = list(zip(paths[index_p],paths[index_p][1:]))

node_col = ['yellow' if node not in paths[index_p] else 'red' for node in G.nodes()]
edge_col = ['black' if edge not in red_edges else 'red' for edge in G.edges()]
edge_labels=dict([((u,v,),d['weight']) for u,v,d in G.edges(data=True)])

pos=nx.random_layout(G)
nx.draw_networkx(G,pos,node_color= node_col,node_size=600)
nx.draw_networkx_edges(G,pos,edge_color= edge_col)
nx.draw_networkx_edge_labels(G,pos,edge_color= edge_col,edge_labels=edge_labels)
pylab.title("Critical Path Duration: "+str(time_t[index_p])+" weeks")
pylab.show()