from src.rl.reward import portfolio_reward
def test_reward_float(): assert isinstance(portfolio_reward(.01), float)
