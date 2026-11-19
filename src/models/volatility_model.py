def ewma_vol(returns, span=21):
    return returns.ewm(span=span).std()*252**0.5
