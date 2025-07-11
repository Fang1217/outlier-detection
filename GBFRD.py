import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.cluster import k_means
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class GBFRD:
    def __init__(self, sigma = 0.6):
        self.sigma = sigma

    def calculate_center_and_radius(self, gb):    
        data_no_label = gb[:,:]
        center = data_no_label.mean(axis=0)
        radius = np.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
        return center, radius

    def splits(self, gb_list, num, k=2):
        gb_list_new = []
        for gb in gb_list:
            p = gb.shape[0]
            if p < num:
                gb_list_new.append(gb)
            else:
                gb_list_new.extend(self.splits_ball(gb, k))
        return gb_list_new

    def splits_ball(self, gb, k):
        ball_list = []
        len_no_label = np.unique(gb, axis=0)
        if len_no_label.shape[0] < k:
            k = len_no_label.shape[0]
        label = k_means(X=gb, n_clusters=k, n_init=1, random_state=8)[1]
        for single_label in range(0, k):
            ball_list.append(gb[label == single_label, :])
        return ball_list

    def assign_points_to_closest_gb(self, data, gb_centers):
        assigned_gb_indices = np.zeros(data.shape[0])
        for idx, sample in enumerate(data):
            t_idx = np.argmin(np.sqrt(np.sum((sample - gb_centers) ** 2, axis=1)))
            assigned_gb_indices[idx] = t_idx
        return assigned_gb_indices.astype('int')

    def fuzzy_similarity(self, t_data, k=2):
        t_n, t_m = t_data.shape
        gb_list = [t_data]
        num = np.ceil(t_n ** 0.5)
        while True:
            ball_number_1 = len(gb_list)
            gb_list = self.splits(gb_list, num=num, k=k)
            ball_number_2 = len(gb_list)
            if ball_number_1 == ball_number_2:
                break
        gb_center = np.zeros((len(gb_list), t_m))
        for idx, gb in enumerate(gb_list):
            gb_center[idx], _ = self.calculate_center_and_radius(gb)
        point_to_gb = self.assign_points_to_closest_gb(t_data, gb_center)
        point_center = np.zeros((t_n, t_m))
        for i in range(t_n):
            point_center[i] = gb_center[point_to_gb[i]]
        tp = 1 - cdist(point_center, point_center) / t_m
        tp[tp < self.sigma] = 0
        return tp

    def calculate_outlier_factor(self, data):
        n, m = data.shape
        LA = np.arange(m)
        weight = np.zeros((n, m), dtype=np.float32)
        Acc_A_a = np.zeros((m, n), dtype=np.float32)
        for idx1, l1 in enumerate(LA):
            rel_mat_k_l, ic = np.unique(self.fuzzy_similarity(data[:,[l1]], k=2), axis=0, return_inverse=True)
            n_items = rel_mat_k_l.shape[0]
            A_d1 = np.setdiff1d(LA, l1)
            rel_mat_P_1 = self.fuzzy_similarity(data[:,A_d1], k=2)
            rel_mat_P = rel_mat_P_1
            rel_mat_P_N = 1 - rel_mat_P
            for i in range(n_items):
                i_tem = np.where(ic == i)[0]
                rel_mat_B = rel_mat_k_l[i]
                low_appr = np.maximum(rel_mat_P_N, rel_mat_B).min(axis=1).sum()
                up_appr = np.minimum(rel_mat_P, rel_mat_B).max(axis=1).sum()
                Acc_A_a[idx1, i_tem] = low_appr / up_appr
                weight[i_tem, idx1] = rel_mat_k_l[i].mean()
        GBOD = np.zeros((n, m))
        for i in range(n):
            for k in range(m):
                GBOD[i, k] = 1 - Acc_A_a[k, i] * weight[i, k]
        GBOF = np.mean(GBOD * (1 - np.sqrt(weight)), axis=1) 
        return GBOF


# if __name__ == '__main__':
#     gbfrd = GBFRD()

#     data = pd.read_csv("./Example.csv").values
#     ID = (data >= 1).all(axis=0) & (data.max(axis=0) != data.min(axis=0))
#     scaler = MinMaxScaler()
#     if any(ID):
#         data[:, ID] = scaler.fit_transform(data[:, ID])
#     out_factors = GBFRD(data)
#     print(out_factors)  

