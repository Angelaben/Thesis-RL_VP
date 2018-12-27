from Environment.BanditEnvironment_stationary import BanditEnvironment

env = BanditEnvironment(5, 8)
assert len(env.list_items) == 8
assert len(env.list_client) == 5
assert len(env.list_item_as_array) == 8
print("=====List item as array=====")
print(env.list_item_as_array)
client, list_as_array = env.reset()
print("Client returned by reset ", client._my_id)
for client in env.list_client:
    print("Client : ",client._my_id, "likes women items " , client._taste[2])

env.step_mono_recommendation(5)