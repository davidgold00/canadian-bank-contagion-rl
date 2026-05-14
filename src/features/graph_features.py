import networkx as nx, pandas as pd

def graph_feature_frame(G, date):
    deg=nx.degree_centrality(G); eig=nx.eigenvector_centrality_numpy(G, weight='weight') if len(G)>1 else {}
    bet=nx.betweenness_centrality(G, weight='weight'); cl=nx.clustering(G, weight='weight')
    return pd.DataFrame({'date':date,'node':list(G.nodes()),'degree':[deg.get(n,0) for n in G.nodes()], 'eigenvector':[eig.get(n,0) for n in G.nodes()], 'betweenness':[bet.get(n,0) for n in G.nodes()], 'clustering':[cl.get(n,0) for n in G.nodes()]})
