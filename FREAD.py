# Fuzzy Rough Entropy-based Anomaly Detection (FREAD) algorithm
# Please refer to the following papers:
# Exploiting fuzzy rough entropy to detect anomalies, International Journal of Approximate Reasoning, 2023.
# Uploaded by Sihan Wang on Nov. 23, 2023. E-mail:wangsihan0713@foxmail.com.
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class FREAD:
    def __init__(self, delta=0.5):
        self.delta = delta

    def getDist(self, Data, attribute_list, category):
        # input:
        # Data is data matrix without decisions, where rows for samples and columns for attributes.
        # Numerical attributes should be normalized into [0,1].
        # Nominal attributes be replaced by different integer values.
        # attribute_list is a list of attribute types for each column in the Data
        # category is a flag to determine whether it is a sequence of attributes or a sequence of attribute subsets.
        if category == 0:  # a sequence of attributes
            if attribute_list == 0:
                return pairwise_distances(Data, metric="hamming")  # nominal
            if attribute_list == 1:
                return pairwise_distances(Data, metric="euclidean")  # numerical

        else:  # a sequence of attribute subsets
            length = len(attribute_list)
            nominal = [index for (index, value) in enumerate(attribute_list) if
                    value == 0]  # Pick the indexes of the nominal attributes
            numerical = [index for (index, value) in enumerate(attribute_list) if
                        value == 1]  # Pick the indexes of the numerical attributes

            nomi_data = Data[:, nominal]  # Take out the nominal data
            num_data = Data[:, numerical]  # Take out the numerical data

            if len(numerical) == length:  # All numerical attributes.
                return pairwise_distances(Data, metric="euclidean")
            elif len(nominal) == length:  # All nominal attributes.
                return pairwise_distances(Data, metric="hamming")
            else:  # hybrid attributes
                dis_nominal = pairwise_distances(nomi_data, metric="hamming")
                dis_numerical = pairwise_distances(num_data, metric="euclidean")
                return dis_nominal + dis_numerical


    def calculate_outlier_factor(self, data):
        # input:
        # data is data matrix without decisions, where rows for samples and columns for attributes.
        # Numerical attributes should be normalized into [0,1].
        # Nominal attributes be replaced by different integer values.
        # delta is a given parameter for calculating the fuzzy similarity relation.

        # step1：Import data
        n, m = data.shape  # Number of rows and columns
        Epsilon = (data <= 1).all(axis=0) & (data.max(axis=0) != data.min(axis=0))  # True = numerical，False = nominal

        # step2：Calculate fuzzy rough entropy and fuzzy rough relative entropy

        # 2.1. Sequence of attribute subsets
        H = np.zeros(m)  # entropy
        H_x = np.zeros((n, m))  # relative entropy
        weight = np.zeros((n, m))  # weight matrix

        for j in range(m):
            R_c = 1 - self.getDist(data[:, j].reshape(-1, 1), Epsilon[j],
                            0)  # Calculate the fuzzy similarity relation between samples under each attribute
            R_c[R_c < self.delta] = 0
            H[j] = sum((1 / n) * np.log2(np.sum(R_c, axis=1)))  # fuzzy rough entropy
            weight[:, j] = ((np.sum(R_c, axis=1)) / n) ** (1 / 3)  # weight

            for i in range(n):
                R_c_te = R_c
                R_c_te = np.delete(R_c_te, i, axis=0)  # Delete the row =[]
                R_c_te = np.delete(R_c_te, i, axis=1)  # Delete the column = []

                H_x[i, j] = sum((1 / (n - 1)) * np.log2(np.sum(R_c_te, axis=1)))  # Fuzzy rough relative entropy of c_j(u_i)

        # 2.2. Sequence of attribute sets
        H_as = np.zeros(m)  # entropy
        H_as_x = np.zeros((n, m))  # relative entropy
        e_as = np.argsort(H)  # In ascending order of entropy.
        data = data[:, e_as]  # Sort by entropy magnitude of each column attribute
        temp = np.array(Epsilon)
        attribute_as_list = temp[e_as]  # The list of attribute types is also sorted by entropy
        weightA_as = np.zeros((n, m))  # weight matrix

        for j in range(m):
            Dis_B = self.getDist(data[:, 0:j + 1], attribute_as_list[0:j + 1], 1)  # distance matrix
            R_B = 1 - Dis_B / (j + 1)
            R_B[R_B < self.delta] = 0
            H_as[j] = sum((1 / n) * np.log2(np.sum(R_B, axis=1)))
            weightA_as[:, j] = (np.sum(R_B, axis=1) / n) ** (1 / 3)

            for i in range(n):
                R_B_te = R_B
                R_B_te = np.delete(R_B_te, i, axis=0)  # Delete the row =[]
                R_B_te = np.delete(R_B_te, i, axis=1)  # Delete the column =[]

                H_as_x[i, j] = sum((1 / (n - 1)) * np.log2(np.sum(R_B_te, axis=1)))

        # step3：Calculate the anomaly scores.

        H_x = H_x[:, e_as]  # Sort by entropy
        H = H[e_as]  # Sort by entropy

        # Calculate the relative entropy matrix of the two sequences according to the relative entropy equation
        H_x = 1 - H_x / H
        H_as_x = 1 - H_as_x / H_as

        H_x[np.isnan(H_x)] = 0
        H_as_x[np.isnan(H_as_x)] = 0

        H_x[H_x > 1] = 1
        H_x[H_x < 0] = 0

        H_as_x[H_as_x > 1] = 1
        H_as_x[H_as_x < 0] = 0

        score = 1 - np.sum((H_x + H_as_x) * (weight + weightA_as) / (4 * m), axis=1)

        return score

# if __name__ == "__main__":
#     load_data = loadmat('FREAD_Example.mat')
#     trandata = load_data['trandata']

#     scaler = Min scaler.fit_transform(trandata[:, 1:])
#     MaxScaler()
#     trandata[:, 1:] =
#     self.delta = 0.5
#     out_scores = FREAD(trandata, self.delta)

#     print(out_scores)
