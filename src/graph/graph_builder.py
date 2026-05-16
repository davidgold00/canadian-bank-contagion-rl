import numpy as np, pandas as pd, networkx as nx
BANKS=["RY.TO","TD.TO","BMO.TO","BNS.TO","CM.TO","NA.TO"]
class DynamicGraphBuilder:
    def __init__(self,banks=None,window=63): self.banks=banks or BANKS; self.window=window
    def adjacency_at(self, returns, date):
        hist=returns.loc[:date,self.banks].tail(self.window).dropna()
        corr=hist.corr().fillna(0).clip(-1,1).abs()
        values = corr.to_numpy(copy=True)
        np.fill_diagonal(values,0)
        return pd.DataFrame(values, index=corr.index, columns=corr.columns)
    def graph_at(self, returns, date, min_weight=.05):
        A=self.adjacency_at(returns,date); G=nx.Graph(); G.add_nodes_from(self.banks)
        for i in self.banks:
            for j in self.banks:
                if i<j and A.loc[i,j]>=min_weight: G.add_edge(i,j,weight=float(A.loc[i,j]))
        return G
