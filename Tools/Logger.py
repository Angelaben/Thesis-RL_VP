import numpy as np
import matplotlib.pyplot as plt
class Logger:
    def __init__(self, n_client, env):
        self.log_reward = []
        self.log_mean_reward = []
        self.log_per_client_reward = [[] for _ in range(n_client)]
        self.log_per_client_reward_mean = [[] for _ in range(n_client)]
        self.log_opti = [[] for _ in range(n_client)]
        print("Logger init : ", np.array(self.log_per_client_reward).shape)
        self.n_client = n_client
        self.env = env


    def add_reward_client(self, reward, client):
        self.log_reward.append(reward)
        self.log_mean_reward.append(np.mean(self.log_reward))
        self.log_per_client_reward[client].append(reward)
        self.log_per_client_reward_mean[client].append(np.mean(self.log_per_client_reward[client]))
        best_buy = self.env.get_indicator()
   #     print("Best buy : ", best_buy)
        for cliend_id in range(self.n_client):
            self.log_opti[cliend_id].append(best_buy[cliend_id])

    def plot(self):

        plt.figure(figsize = (5, 5))
        plt.plot(self.log_mean_reward, label = "Mean reward")
        plt.grid()
        plt.legend()
        plt.figure(figsize = (5, 5))
        for index_client in range(self.n_client):
            plt.plot(self.log_per_client_reward_mean[index_client], label = "Client {}".format(index_client))

        plt.grid()
        plt.legend()
        plt.show()

    def plot_opti_per_client(self):
        plt.subplot(111)
        axis = plt.gca()
        axis.set_ylim([0, 1])
        #plt.figure(figsize = (5, 5))
        for client_index in range(self.n_client):
            plt.scatter(self.log_opti[client_index], label = "Client {}".format(client_index))
        plt.legend()
        plt.show()


