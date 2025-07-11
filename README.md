# Rough Set Theory in Outlier Detection

## Abstract

Outlier detection is a process that detects patterns or data points in a dataset that are deviated from expectations or normal. It is applied in various fields such as medical field in cancer detection, e-commerce fraud detection, network intrusion and many more. Rough Set Theory, introduced by Pawlak in 1982, is a mathematical framework that deals with uncertainty. It is widely applied by researchers in outlier detection. This project presents a comprehensive comparison of three rough outlier detection algorithms, those are Fuzzy Rough Entropy-based Anomaly Detection (FREAD), Granular-ball Fuzzy Rough Sets-Based Anomaly Detection (GBFRD) and a proposed algorithm that is based on GBFRD that is Granular Radial Basis Fuzzy Rough Function-Based Anomaly Detection. The algorithms are compared using twelve publicly available datasets that are designed to evaluate outlier detection algorithms, comprising numerical, categorical and mixed data attributes. The performance of these algorithms is evaluated using three evaluation metrics, including AOC-ROC, AOC-PR, execution time. Experiment results shows that the proposed algorithm GBFRD-RBF achieve the lowest execution time across all algorithms, while performing like GBFRD, where both exceeds FREAD in AUC-ROC and AUC-PR. 

## Dataset
There are several publicly available datasets that contains nominal, numerical, and mixed datasets. The dataset repository to be used in the project will be from https://github.com/BELLoney/Outlier-detection. The details of the datasets are provided in below table:

| **Type** | **Dataset** | **Preprocessing from original URI dataset** | **Numerical** | **Categorical** | **Objects** | **Outliers (%)** |
| --- | --- | --- | --- | --- | --- | --- |
| Nominal | “Audiology_variant1” | Combined classes "cochlear_age", "cochlear_age_and_noise", "cochlear_poss_noise","cochlear_unknown", and "normal_ear" as inliers, other classes as outliers. | 0   | 69  | 226 | 53 (23.45%) |
| Nominal | “Nursery_variant1” | Classes "recommend" and "very_recom" as outliers otherwise as inliers | 0   | 8   | 12,960 | 330 (2.55%) |
| Nominal | “Lymphography” | ''1'' and ''4'' are categorised as outliers | 0   | 8   | 148 | 6 (4.05%) |
| Nominal | “Mushroom_p_573_variant1” | Number of objects of class ''+" is reduced to 573 | 0   | 22  | 4,781 | 573 (11.98%) |
| Numerical | “Yeast_ERL_5_variant1” | Selected “ERL” (outlier), “CYT”, “NUC”, and “MIT” classes | 8   | 0   | 1,141 | 5 (0.44%) |
| Numerical | “Wisconsin_malignant_39_variant1” | Removed 202 “malignant” and 14 “benign” objects | 9   | 0   | 483 | 39 (8.07%) |
| Numerical | “Cardiotocography_2and3_3_variant1” | Number of objects of class “2” and “3” is reduced to 33 | 21  | 0   | 1,688 | 33 (1.95%) |
| Numerical | “Diabetes_tested_positive_26_variant1” | Number of objects of class “tested_positive” is reduced to 26 | 8   | 0   | 526 | 26 (4.94%) |
| Mixed | “Heart_2_16_variant1” | Number of objects of class “2” is reduced to 16 | 6   | 7   | 166 | 16 (9.64%) |
| Mixed | “CreditA_plus_42_variant1” | Number of objects of class “+” is reduced to 42 | 6   | 9   | 425 | 42 (9.88%) |
| Mixed | “Arrhythmia_variant1” | Class “3”, “4”, “5”, “7”, “8”, “9”, “14”, “15” are combined as outliers. Others are combined as inliers. | 206 | 73  | 452 | 66 (14.60%) |
| Mixed | “Sick_sick_72_variant1” | Number of objects of class “sick” is reduced to 72 | 7   | 22  | 3,612 | 72 (1.99%) |


## Usage

There are some requirements that are needed to be installed, first install the requirements:

```
pip install -r requirements.txt
```

To reproduce the usage of the paper, simply run the `__init__.py` file
```
./__init__.py
```

