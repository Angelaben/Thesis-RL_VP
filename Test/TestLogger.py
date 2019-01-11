from Tools.Logger import Logger
from Environment.BanditEnvironment_stationary import BanditEnvironment
import numpy as np
env = BanditEnvironment(5, 5)

logger = Logger(5, env)
res = np.array(logger.log_per_client_mean_best_buy)
rew_mean = np.array(logger.log_per_client_reward_mean)
print(res)
print(rew_mean)
logger.add_reward_client(1, 1, None)
res = np.array(logger.log_per_client_mean_best_buy)
rew_mean = np.array(logger.log_per_client_reward_mean)
print(res)
print(rew_mean)
logger.add_reward_client(1, 1, None)
res = np.array(logger.log_per_client_mean_best_buy)
rew_mean = np.array(logger.log_per_client_reward_mean)
print(res)
print(rew_mean)