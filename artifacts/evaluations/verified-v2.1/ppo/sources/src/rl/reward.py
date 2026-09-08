import numpy as np

def portfolio_reward(portfolio_return, vol=0, drawdown=0, turnover=0, contagion=0, stress=0, excess=0, cfg=None):
    cfg=cfg or {'lambda_vol':.2,'lambda_dd':.3,'lambda_turnover':.02,'lambda_contagion':.15,'lambda_stress':.1,'lambda_alpha':.25}
    return float(portfolio_return - cfg['lambda_vol']*vol - cfg['lambda_dd']*abs(drawdown) - cfg['lambda_turnover']*turnover - cfg['lambda_contagion']*contagion/100 - cfg['lambda_stress']*stress/100 + cfg['lambda_alpha']*excess)
