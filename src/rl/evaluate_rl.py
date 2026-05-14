def run_policy(env, model=None):
    obs,_=env.reset(); rows=[]; done=False
    while not done:
        action = model.predict(obs, deterministic=True)[0] if model else env.action_space.sample()
        obs,r,done,trunc,info=env.step(action); rows.append({'reward':r, **{k:v for k,v in info.items() if k!='weights'}})
    return rows
