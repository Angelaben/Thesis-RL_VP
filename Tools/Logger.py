import numpy as np
import matplotlib.pyplot as plt
class Logger:
    def __init__(self, n_client, env, lim = 100):
        self.log_reward = []
        self.log_mean_reward = []
        self.log_per_client_reward = [[] for _ in range(n_client)]
        self.log_per_client_reward_mean = [[] for _ in range(n_client)]
        self.log_opti = [[] for _ in range(n_client)]
        self.log_mean_opti = []
        self.log_random = []
        self.log_random_mean = []
        print("Logger init : ", np.array(self.log_per_client_reward).shape)
        self.n_client = n_client
        self.env = env
        self.lim = lim


    def add_reward_client(self, reward, client, reward_random = None):
        self.log_reward.append(reward)
        self.log_mean_reward.append(np.mean(self.log_reward[-self.lim:]))
        self.log_per_client_reward[client].append(reward)
        self.log_per_client_reward_mean[client].append(np.mean(self.log_per_client_reward[client][-self.lim:]))
        best_buy = self.env.get_indicator()
   #     print("Best buy : ", best_buy)
        for cliend_id in range(self.n_client):
            self.log_opti[cliend_id].append(best_buy[cliend_id])
        self.log_mean_opti.append(np.mean(best_buy)) # La ligne opti est la moyenne des best buy, dans la mesure
        # ou chaque client est uniformement repartit
        if reward_random is not None:
            self.log_random.append(reward_random)
            self.log_random_mean.append(np.mean(self.log_random[-self.lim:]))

    def plot(self):
        plt.figure(1, figsize = (5, 5))
        plt.axis([0, min(len(self.log_mean_reward), self.lim), 0, 0.4])
        plt.plot(self.log_mean_reward[-self.lim:], label = "Mean reward")
        plt.plot(self.log_mean_opti[-self.lim:], label = "Mean ref optimal")
        if len(self.log_random_mean) > 0:
            plt.plot(self.log_random_mean[-self.lim:], label = "Mean random")
        plt.grid()
        plt.legend()
        plt.figure(2, figsize = (5, 5))
       # plt.axis([0, len(self.log_per_client_reward_mean), 0, 2])
        for index_client in range(self.n_client):
            plt.plot(self.log_per_client_reward_mean[index_client][-self.lim:], label = "Client {}".format(index_client))
        plt.plot(self.log_mean_opti[-self.lim:], label = "Mean ref optimal")
        plt.grid()
        plt.legend()
        plt.show()
        self.plot_opti_per_client()

    def plot_opti_per_client(self):
        plt.figure(3, figsize = (5, 5))
        plt.axis([0, 10000, 0, 1])
        for client_index in range(self.n_client):
            plt.plot(self.log_opti[client_index], label = "Client indicator {}".format(client_index))
        plt.legend()
        plt.show()


