from Environment.Client import Client
import matplotlib.pyplot as plt
import numpy as np
# Test repartition valeur de generate normal
cl = Client(0, 5)
res = cl.generate_normal_distirbution(0, 100, 50, 10)
plt.plot(range(100), res)
plt.show()
print(np.sum(res))

res = cl.generate_normal_distirbution(20 - 2, 100, 20, 5) # Simuler l'idee qu'un client ne s'interesse qu a un age restreint autour de lui
plt.plot(range(82), res)
plt.show()