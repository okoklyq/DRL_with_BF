import ctypes
import random

import numpy
import numpy as np
import copy
import math

from gym.vector.utils import spaces
from sklearn import preprocessing


class Environment:
    def __init__(self, k, N, noise_power, state_dim, action_dim, beams):
        self.k = k
        self.C = beams  # number of clusters/beams
        self.N = N
        self.B = 1 * k  # Mhz
        self.T = 50
        self.power_unit = 1000 # 最大发射功率 1000mW,30dbm
        self.noise_power = noise_power

        self.H_ak = np.random.normal(scale=0.5, size=(self.N, self.k, self.T)) + np.random.normal(scale=0.5, size=(
        self.N, self.k, self.T)) * 1j
        self.P_ap = np.array([212, 212, 1])  # position
        self.P_k_list = np.random.normal(scale=10, size=(self.N, self.k))
        self.P_k_list[2, :] = 0
        self.P_k_list_initial = copy.deepcopy(self.P_k_list)
        self.P_k = 0
        self.t = 0
        self.Rice = 1

        self.rate_qos = 0.5
        self.qos = True

        self.los_ak = np.ones(self.N) + 0 * 1j
        self.r_ak_list = np.zeros(shape=(self.k, self.N)) + 0 * 1j
        # self.w_ak_list = np.zeros(shape=(self.N, self.N)) + 0 * 1j # 有ZF时
        self.w_ak_list = np.ones(shape=(self.C, self.N)) + 0 * 1j  # self.C is the number of beams
        self.ee_list = []
        self.ee_sum = 0
        self.sum_rate = 0
        self.sum_sinr = 0
        self.avg_sinr = 0
        self.clustering_log = ''
        self.I_intra_cluster = [0] * self.k
        self.I_inter_cluster = [0] * self.k

        self.h_list = np.zeros(self.k)
        self.power_allocation_list = np.ones(self.k) * self.power_unit
        self.power_for_each_beam_list = np.zeros(self.C)
        self.data_rate_list = np.zeros(self.k)
        self.SINR_list = np.zeros(self.k)
        self.s = 0

        self.num_states = state_dim
        self.action_dim = action_dim

        # self.alpha_list = []
        # self.power_factor = [1] * self.C  # decided by agent 每一个元素的值应该 ∈【1，C】
        self.power_for_user_list = [0] * self.k


        self.ep_reward_list = []

    def calculate_loss(self):
        for k in range(self.k):
            self.P_k = self.P_k_list[:, k].T
            d_ak = np.linalg.norm(self.P_k - self.P_ap)
            if d_ak < 1:
                d_ak = 1
            PL_ak = 10 ** (-30 / 10) * (d_ak ** (-3.5))  # alpha_au = 3.5 PL_au单位db
            Nlos_ak = self.H_ak[:, k, self.t]
            los_ak = self.los_ak
            r_ak = np.sqrt(PL_ak) * (np.sqrt(self.Rice / (1 + self.Rice)) * los_ak + np.sqrt(1 / (1 + self.Rice)) * Nlos_ak)  # the channel, h_k[n] in paper
            # print("user: ", k, "r_ak: ", r_ak)
            self.r_ak_list[k] = r_ak
        # print("r_ak_list： ", self.r_ak_list)

    def calculate_data_rate(self):
        # power allocation

        # power for each user
        for k in range(self.k):
            self.power_allocation_list[k] = self.power_unit * self.beta * self.power_for_user_list[k]
            # print("user: ", k, "allocated power: ", self.power_allocation_list[k])

        # power for each beam
        for c in range(self.C):
            for k in range(self.k):
                if c == self.clustering_result[k]:
                    self.power_for_each_beam_list[c] += self.power_allocation_list[k]
            # print("beam: ", c, "power: ", self.power_for_each_beam_list[c])

        # for c in range(self.C):
        #     self.power_for_each_beam_list[c] = self.power_for_each_beam_list[c] * self.power_factor[c] * self.beta
        #     # print("beam: ", c, "power: ", self.power_for_each_beam_list[c])
        #     self.w_ak_list[c] = self.w_ak_list[c] * self.power_for_each_beam_list[c]
        #
        # for c in range(self.C):
        #     for i in range(self.k):
        #         if c == self.clustering_result[i]:
        #             self.power_allocation_list[i] = self.power_for_each_beam_list[c] * self.alpha_list[i]
        #             # print("user: ", i, "allocated power: ", self.power_allocation_list[i])

        # beamforming
        for k in range(self.k):
            beam = self.clustering_result[k]
            g_k_H = abs(np.dot(self.r_ak_list[k], self.w_ak_list[beam])) ** 2  # |h_k * w_k|^2
            # print("gkh:", g_k_H)
            self.h_list[k] = g_k_H
        # print("h_list: ", self.h_list)

        # interference
        for i in range(self.k):
            I_intra_interference = 0
            I_inter_interference = 0
            for j in range(self.k):
                if self.clustering_result[i] == self.clustering_result[j]:  # in the same cluster
                    if i != j and self.h_list[j] > self.h_list[i]:
                        I_intra_interference += self.power_allocation_list[j] * self.h_list[i]
                else:  # the two are in different clusters
                        I_inter_interference += self.power_allocation_list[j] * abs(np.dot(self.r_ak_list[i],self.w_ak_list[self.clustering_result[j]])) ** 2  # self.h_list[i]
            self.I_inter_cluster[i] = I_inter_interference
            self.I_intra_cluster[i] = I_intra_interference
            # print("user: ", i, "intra_cluster: ", I_intra_interference, "inter_cluster: ", I_inter_interference)

        # data rate
        for k in range(self.k):
            Signal_power_NOMA_k = self.power_allocation_list[k] * self.h_list[k]
            # print("user: ", k, "signal: ", Signal_power_NOMA_k)
            SINR_NOMA_k = Signal_power_NOMA_k / (self.I_intra_cluster[k] + self.I_inter_cluster[k] + self.noise_power)
            # print("user: ", k, "SINR: ", SINR_NOMA_k)
            data_rate_k = self.B * math.log((1 + SINR_NOMA_k), 2)
            # print("user: ", k, "data rate:", data_rate_k)

            self.data_rate_list[k] = data_rate_k
            self.SINR_list[k] = SINR_NOMA_k

        # qos
        for k in range(self.k):
            if self.data_rate_list[k] < self.rate_qos:
                self.qos = False

    def calculate_energy_efficiency(self):
        # rate / power
        sum_rate = sum(self.data_rate_list)
        sum_power = sum(self.power_allocation_list)
        sum_sinr = sum(self.SINR_list)

        # for k in range(self.k):
        #     sum_rate += self.data_rate_list[k]
        #     sum_power += self.power_allocation_list[k]
        #     sum_sinr += self.SINR_list[k]  # to max, min, average one
        self.transmitting_power = self.power_unit * self.beta
        ee_sum = sum_rate / sum_power
        # print("sum rate: ", sum_rate, "power: ", sum_power, "ee: ", ee_sum)

        self.sum_rate = sum_rate
        self.sum_power = sum_power
        self.sum_sinr = sum_sinr
        self.avg_sinr = self.sum_sinr/self.k
        self.ee_sum = ee_sum
        self.ee_list.append(ee_sum)

        # print("sum: ", self.sum_power, self.sum_rate, self.sum_sinr)

        # reset after each step
        self.power_allocation_list = np.ones(self.k) * self.power_unit
        self.power_for_each_beam_list = np.zeros(self.C)

        self.I_intra_cluster = [0] * self.k
        self.I_inter_cluster = [0] * self.k

    def user_move(self):
        self.P_k_list[0, :] = self.P_k_list[0, :] + np.random.normal(scale=0.1, size=(1, self.k))
        self.P_k_list[1, :] = self.P_k_list[1, :] + np.random.normal(scale=0.1, size=(1, self.k))

    def get_state(self):
        state = np.ndarray([self.num_states, ])
        h_list_S = np.append(np.real(self.r_ak_list), np.imag(self.r_ak_list))
        state[0:] = h_list_S
        return state

    def step(self, action):
        action = action + 1
        for x in range(self.action_dim):
            if action[x] < 0:
                action[x] = -action[x]

        # beamforming vector
        angle_list = action[0:self.C*self.N] * 2 * math.pi
        amplitude_list = action[self.C*self.N:2*self.C*self.N]
        w_ak_list = np.zeros(shape=(1, self.C*self.N)) + 0*1j
        w_ak_sum = 0

        cos_list = [0]*(self.C*self.N)
        sin_list = [0]*(self.C*self.N)

        for a in range(self.C*self.N):
            x = angle_list[a]
            cos = math.cos(x) * amplitude_list[a]
            sin = math.sin(x) * amplitude_list[a]

            cos_list[a] = cos
            sin_list[a] = sin

        for a in range(self.C*self.N):
            w_ak_list[0,a] = cos_list[a] + sin_list[a]*1j

        # 归一化
        for i in range(self.C * self.N):
            w_ak_sum +=  cos_list[i]**2 + sin_list[i]**2
        w_ak_sum = math.sqrt(w_ak_sum)

        w_ak_list = w_ak_list/w_ak_sum

        w_ak_list = np.reshape(w_ak_list, [self.C, self.N])
        # print("w_ak_liat: ", self.w_ak_list)

        # clustering result
        # the original one!
        original_data = action[2*self.C*self.N:2*self.C*self.N+self.k]
        scaled_data = preprocessing.minmax_scale(original_data)
        clustering_result = [int(cluster * (self.C-1)) for cluster in scaled_data]
        clustering_log = [str(x) for x in clustering_result]
        # print("clustering result: ", clustering_result)

        # # test
        # original_data = action[self.C * self.N:self.C * self.N + self.k]
        # clustering_result = [int((cluster/sum(original_data)) * self.C) for cluster in original_data]
        # clustering_log = [str(x) for x in clustering_result]

        # # alpha list
        # alpha_list = [0] * self.k
        # original_alpha = action[self.C*self.N+self.k:self.C*self.N+self.k+self.k]
        # # print("check original alpha: ", original_alpha)
        #
        # for i in range(self.k):
        #     term_list = np.zeros(self.k)
        #     term_user_list = []
        #     term_list[i] = original_alpha[i]
        #     term_user_list.append(i)
        #     for j in range(self.k):
        #         if i != j and clustering_result[i] == clustering_result[j]:  # in the same cluster
        #                 term_list[j] = original_alpha[j]
        #                 term_user_list.append(j)
        #     for x in term_user_list:
        #         alpha_list[x] = term_list[x]/sum(term_list)
        # # print("alpha list: ", alpha_list)
        #
        # # power factor _ power for beams
        # power_factor_list = [0] * self.C
        # original_factor = action[self.C*self.N+self.k+self.k:self.C*self.N+self.k+self.k+self.C]
        # for beam in range(self.C):
        #     power_factor_list[beam] = original_factor[beam]/sum(original_factor)
        # # print("power_factor_list: ", power_factor_list)

        # allocated power for each user
        power_for_user_list = [0] * self.k
        original_list = action[2*self.C*self.N+self.k:2*self.C*self.N+self.k+self.k]
        for user in range(self.k):
            power_for_user_list[user] = original_list[user]/sum(original_list)
        # print("power_for_user_list: ", power_for_user_list)

        # beta
        self.beta = action[2*self.C*self.N+self.k+self.k]
        while self.beta < 0.7 or self.beta > 1:
            if self.beta > 2:
                self.beta = random.uniform(0.9, 1)
            if self.beta <= 2 and self.beta > 1:
                self.beta = random.uniform(0.8, 0.9)
            if self.beta < 0.7:
                self.beta = random.uniform(0.7, 0.8)
        # print("beta: ", self.beta)

        self.w_ak_list = w_ak_list
        # self.alpha_list = alpha_list
        self.clustering_result = clustering_result
        self.clustering_log = "_".join(clustering_log)
        self.power_for_user_list = power_for_user_list
        # self.power_factor = power_factor_list

        self.calculate_data_rate()
        self.calculate_energy_efficiency()

        if self.qos == False:
            reward = 0.8 * self.ee_sum
        else:
            reward = self.ee_sum

        self.H_ak = np.random.normal(scale=0.5, size=(self.N, self.k, self.T)) + np.random.normal(scale=0.5, size=(
            self.N, self.k, self.T)) * 1j
        self.calculate_loss()
        return self.get_state(), reward, False, {}

    def reset(self):
        self.P_k_list = copy.deepcopy(self.P_k_list_initial)
        self.t = 0







