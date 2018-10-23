from od_mstar3 import workspace_graph
import networkx as nx #Python network analysis module



def to_networkx_graph(obs_map):
    '''Reads in a standard obs_map list and converts it to a networkx
    digraph
    obs_map - list of lists, 0 for empty cell, 1 for obstacle'''
    #Create a workspace_graph object to generate neighbors
    g = workspace_graph.Astar_Graph(obs_map,[0,0])
    G = nx.DiGraph() #Creates the graph object
    #Populate graph with nodes
    for x in range(len(obs_map)):
        for y in range(len(obs_map[x])):
            if obs_map[x][y] == 0:
                G.add_node((x,y))
    #Add edges
    for i in G.nodes():
        #Stored nodes by their coordinates in G
        for j in g.get_neighbors(i):
            G.add_edge(i,j)
    return G

def prune_opposing_edge(G,num_edges=1):
    '''Reads in a networkx digraph and prunes the edge opposing the most
    between (i.e. edge on the most shortest path connections).  If this edge
    doesn't have an opposing edge, or if the removal of said edge would
    reduce the connectivity of the space, the next most between edge is pruned
    instead.  Since computing completeness can be expensive, allows multiple
    edges to be pruned before computing the impact of said prunning on
    completeness is computed'''
    #Get the current number of strongly connected components, can't decrease
    #without preventing some paths from being found
    num_components = nx.number_strongly_connected_components(G)
    pruned = 0
    # print 'computing betweeness'
    betweenness = nx.edge_betweenness_centrality(G)
    # print 'betweenness computed'
    while pruned < num_edges:
        max_bet = max(betweenness.values())
        if max_bet <= 0:
            #Set betweeness to -1 if can't prune, set to 0 not between
            return G
        edge = betweenness.keys()[betweenness.values().index(max_bet)]
        if not (edge[1],edge[0]) in G.edges():
            #Already been pruned
            betweenness[edge] = -1
            # print 'no edge'
            continue
        #Test if pruning the edge will break connectivity
        temp_graph = G.copy()
        temp_graph.remove_edge(edge[1],edge[0])
        if num_components == nx.number_strongly_connected_components(temp_graph):
            #Can safely prune this edge
            G = temp_graph
            pruned+=1
            betweenness[edge] = -1
            betweenness.pop((edge[1],edge[0]))
            # print 'pruned'
            #Need to prevent further edges from being pruned from this vertex
            for neighbor in G.neighbors(edge[1]):
                betweenness[(edge[1],neighbor)] = -1
        else:
            betweenness[edge] = -1
            # print 'breaks con %s' %(str(edge))
    return G
