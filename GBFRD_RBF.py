import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.cluster import k_means
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class GBFRD_RBF:
    def __init__(self, sigma=0.6):
        self.sigma = sigma  # sigma now serves as the RBF bandwidth parameter

    def calculate_center_and_radius(self, gb):    
        data_no_label = gb[:, :]
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
        unique_data = np.unique(gb, axis=0)
        if unique_data.shape[0] < k:
            k = unique_data.shape[0]
        label = k_means(X=gb, n_clusters=k, n_init=1, random_state=8)[1]
        for single_label in range(k):
            ball_list.append(gb[label == single_label, :])
        return ball_list

    def assign_points_to_closest_gb(self, data, gb_centers):
        distances = cdist(data, gb_centers)
        assigned_gb_indices = np.argmin(distances, axis=1)
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
        gb_centers = np.array([self.calculate_center_and_radius(gb)[0] for gb in gb_list])
        point_to_gb = self.assign_points_to_closest_gb(t_data, gb_centers)
        point_centers = gb_centers[point_to_gb]
        # Compute RBF kernel
        squared_distances = cdist(point_centers, point_centers, 'sqeuclidean')
        if self.sigma == 0:
            gamma = 0
        else:
            gamma = 1.0 / (2 * (self.sigma ** 2))
        similarity_matrix = np.exp(-gamma * squared_distances)
        return similarity_matrix

    def calculate_outlier_factor(self, data):
        n, m = data.shape
        features = np.arange(m)
        weight = np.zeros((n, m), dtype=np.float32)
        Acc_A_a = np.zeros((m, n), dtype=np.float32)
        
        for feature_idx in features:
            # Similarity matrix for the current single feature
            feature_similarity = self.fuzzy_similarity(data[:, [feature_idx]], k=2)
            # Process remaining features
            other_features = np.setdiff1d(features, feature_idx)
            other_features_similarity = self.fuzzy_similarity(data[:, other_features], k=2)
            other_features_similarity_complement = 1 - other_features_similarity
            
            # Calculate approximations
            for i in range(n):
                similarity_value = feature_similarity[i, i]  # Using self-similarity as representative
                lower_approx = np.maximum(other_features_similarity_complement[i], similarity_value).min()
                upper_approx = np.minimum(other_features_similarity[i], similarity_value).max()
                Acc_A_a[feature_idx, i] = lower_approx / upper_approx if upper_approx != 0 else 0
                weight[i, feature_idx] = feature_similarity[i].mean()
        
        # Calculate outlier factors
        GBOD = 1 - Acc_A_a.T * weight
        GBOF = np.mean(GBOD * (1 - np.sqrt(weight)), axis=1)
        return GBOF
