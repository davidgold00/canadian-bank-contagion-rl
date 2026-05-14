class RollingGraphDataset:
    def __init__(self, returns, builder): self.returns=returns; self.builder=builder
    def __iter__(self):
        for date in self.returns.index[self.builder.window:]: yield date, self.builder.adjacency_at(self.returns,date)
