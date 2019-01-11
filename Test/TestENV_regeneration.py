from Environment.BanditEnvironment_stationary_WithRegeneration import BanditEnvironment_regeneration

env = BanditEnvironment_regeneration()
res = env.reset()
print(res)
env.step_mono_recommendation(0)
first_item = res[1][0]
env.regenerate()
res = env.reset()
regenerated_first_item = res[1][0]
assert first_item.get_Price != regenerated_first_item.get_Price # Might fail at random if same price
# Mais si experience repete ne doit pas fail