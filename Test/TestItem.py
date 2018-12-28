from Environment import Items
item = Items.Items(5, 120, 42)

assert item.get_Price < 120
assert item.get_color < 42
assert item.get_id == 5
print("List item ", item.list_item())
print("Get as list ", item.get_as_list())
print("Properties ", item.get_properties)