from Environment.BanditEnvironment_stationary import BanditEnvironment
n_item = 4
env = BanditEnvironment(n_client = 5, n_item = n_item)
assert len(env.list_items) == n_item
assert len(env.list_client) == 5
assert len(env.list_item_as_array) == n_item
print("=====List item as array=====")
print(env.list_item_as_array)
client, list_as_array = env.reset()
print("Client returned by reset ", client)
for client in env.list_client:
    print("Client : ",client._my_id, "likes women items " , client._taste[2])

for i in range(10):
    env.step_mono_recommendation(n_item - 1)
    print("Indicator : ", env.get_indicator())