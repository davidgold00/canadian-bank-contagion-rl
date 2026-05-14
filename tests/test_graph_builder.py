from src.data.sample_data import make_sample_market_data
from src.graph.graph_builder import DynamicGraphBuilder, BANKS

def test_adjacency_shape(tmp_path):
    p=make_sample_market_data(tmp_path/'p.csv', n=100); r=p.pct_change(); A=DynamicGraphBuilder().adjacency_at(r, r.index[-1]); assert A.shape==(len(BANKS),len(BANKS))
