from GBFRD import GBFRD 
from FREAD import FREAD 
from GBFRD_RBF import GBFRD_RBF
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score


# .mat files from /Dataset folder used
datasets_mixed = ["Arrhythmia", "Credit_A_Plus", "Heart", "Sick"]
datasets_nominal = ["Audiology", "Lymphography", "Mushroom", "Nursery"]
datasets_numerical = ["Cardiotocography", "Diabetes", "WDBC", "Yeast"]
all_datasets = datasets_mixed + datasets_nominal + datasets_numerical
# Uncomment the line below to use only Sick dataset for testing
# all_datasets = ["Sick"]

dataset_to_location = {
    "Arrhythmia":       "./Dataset/Mixed/arrhythmia_variant1.mat", 
    "Credit_A_Plus":    "./Dataset/Mixed/creditA_plus_42_variant1.mat",
    "Heart":            "./Dataset/Mixed/heart270_2_16_variant1.mat", 
    "Sick":             "./Dataset/Mixed/sick_sick_72_variant1.mat",
    "Audiology":        "./Dataset/Nominal/audiology_variant1.mat",
    "Lymphography":     "./Dataset/Nominal/lymphography.mat",
    "Mushroom":         "./Dataset/Nominal/mushroom_p_573_variant1.mat",
    "Nursery":          "./Dataset/Nominal/nursery_variant1.mat",
    "Cardiotocography": "./Dataset/Numerical/cardiotocography_2and3_33_variant1.mat",
    "Diabetes":         "./Dataset/Numerical/diabetes_tested_positive_26_variant1.mat",
    "WDBC":             "./Dataset/Numerical/wdbc_M_39_variant1.mat",
    "Yeast":            "./Dataset/Numerical/yeast_ERL_5_variant1.mat"
}

def read_dataset(dataset):
    dataset_location = dataset_to_location[dataset]
    load_data = loadmat(dataset_location)
    data = load_data['trandata']

    # Last column is extracted as Decision Column from dataset, and then removed from the dataset.
    decision_column = data[:, -1]
    data = data[:, :-1]

    # Perform MinMax Scaling on the dataset to normalize data
    # Note: This is already done on the dataset.
    # column_is_attribute = (data >= 1).all(axis=0) & (data.max(axis=0) != data.min(axis=0))
    # scaler = MinMaxScaler()
    # if any(column_is_attribute):
    #    data[:, column_is_attribute] = scaler.fit_transform(data[:, column_is_attribute])

    return data, decision_column


def save_to_excel(data, filename):
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)  

def plot_auc_roc(decision_column, outlier_scores, algorithm_name, dataset, parameter_value):
    """
    Plots the ROC curve and displays the AUC-ROC value.
    
    Parameters:
        decision_column (array-like): True labels (0 for normal, 1 for anomalies).
        outlier_scores (array-like): Anomaly scores predicted by the algorithm.
        algorithm_name (str): Name of the algorithm used.
        dataset (str): Name of the dataset used.
        parameter_value (float): Value of the parameter used in the algorithm.
    """
    # Calculate FPR, TPR, and thresholds
    fpr, tpr, _ = roc_curve(decision_column, outlier_scores)
    auc = roc_auc_score(decision_column, outlier_scores)

    # Calculate the detection rate
    
    detection_rate = (tpr + fpr) / len(outlier_scores) if (len(outlier_scores)) > 0 else 0
    print(f"Detection Rate: {detection_rate}")

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {algorithm_name}')
    plt.legend(loc='lower right')
    plt.grid()

    # Output the graph to a file
    plt.savefig(f'Output/ROC_{algorithm_name}_{dataset}_{parameter_value:.2f}.png')
    plt.close()  # Close the plot to free memory

def plot_auc_pr(decision_column, outlier_scores, algorithm_name, dataset, parameter_value):
    """
    Plots the Precision-Recall curve and displays the AUC-PR value.
    
    Parameters:
        decision_column (array-like): True labels (0 for normal, 1 for anomalies).
        outlier_scores (array-like): Anomaly scores predicted by the algorithm.
        algorithm_name (str): Name of the algorithm used.
        dataset (str): Name of the dataset used.
        parameter_value (float): Value of the parameter used in the algorithm.
    """
    # Calculate precision, recall, and thresholds
    precision, recall, _ = precision_recall_curve(decision_column, outlier_scores)
    auc_pr = auc(recall, precision)

    # Plot the Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, color='blue', label=f'AUC-PR = {auc_pr:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve for {algorithm_name}')
    plt.legend(loc='lower right')
    plt.grid()

    # Save the plot to a file
    plt.savefig(f'Output/PR_{algorithm_name}_{dataset}_{parameter_value:.2f}.png')    
    plt.close()  # Close the plot to free memory



if __name__ == '__main__':
    # GBFRD
    df_auc_roc = pd.DataFrame(columns=['Dataset'] + [f'{(i * 0.05):.2f}' for i in range(0, 21)])
    df_auc_pr = pd.DataFrame(columns=['Dataset'] + [f'{(i * 0.05):.2f}' for i in range(0, 21)])
    df_time = pd.DataFrame(columns=['Dataset'] + [f'{(i * 0.05):2f}' for i in range(0, 21)])

    for dataset in all_datasets:
        data, decision_column = read_dataset(dataset=dataset)

        # Sigma is set from 0 to 1 with increment of 0.05
        sigma = 0.0

        time_history = []
        auc_roc_history = []
        auc_pr_history = []

        while sigma <= 1.0:
            start_time = time.time()
            print(f"--- Starting GBFRD ({dataset} dataset) with sigma = {sigma} ---")

            gbfrd = GBFRD(sigma=sigma)
            out_factors = gbfrd.calculate_outlier_factor(data)
            # print(f"Sigma: {sigma}, Outlier Factors: {out_factors}")

            # Calculate execution time
            elapsed_time = time.time() - start_time
            print("--- Completed: %s seconds ---" % (elapsed_time))
            time_history += [round(elapsed_time, 4)]
            # print(out_factors)  

            # Calculate detection rate of outliers
            number_of_outliers = np.sum(decision_column == 1)
            number_of_outliers_detected = np.sum(out_factors)
            number_of_outliers_correctly_detected = np.sum(out_factors[decision_column == 1])

            true_positive = np.sum(out_factors[decision_column == 1])
            true_negative = np.sum(out_factors[decision_column == 0])
            false_positive = np.sum(out_factors[decision_column == 0])
            rate = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            print(f"Detection Rate: {rate}")
            
            detection_rate = np.sum(out_factors) / len(out_factors)
            print(f"Detection Rate: {detection_rate}")

            # Calculate AUC for GBFRD
            auc_roc_gbfrd = roc_auc_score(decision_column, out_factors)

            precision, recall, _ = precision_recall_curve(decision_column, out_factors)
            auc_pr_gbfrd = auc(recall, precision)
            print(f"GBFRD AUC-ROC: {auc_roc_gbfrd}")
            print(f"GBFRD AUC-PR: {auc_pr_gbfrd}")
            auc_roc_history += [round(auc_roc_gbfrd, 4)]
            auc_pr_history += [round(auc_pr_gbfrd, 4)]

            plot_auc_roc(decision_column, out_factors, "GBFRD", dataset, sigma)
            plot_auc_pr(decision_column, out_factors, "GBFRD", dataset, sigma)

            sigma = round(sigma + 0.05, 2)

        df_auc_roc.loc[len(df_auc_roc)] = [dataset] + auc_roc_history
        df_auc_pr.loc[len(df_auc_pr)] = [dataset] + auc_pr_history
        df_time.loc[len(df_time)] = [dataset] + time_history

    df_auc_roc.to_excel('Output/ROC_GBFRD.xlsx', index=False)
    df_auc_pr.to_excel('Output/PR_GBFRD.xlsx', index=False)
    df_time.to_excel('Output/ET_GBFRD.xlsx', index=False)



    # GBFRD with RBF

    df_auc_roc = pd.DataFrame(columns=['Dataset'] + [f'{(i * 0.05):.2f}' for i in range(0, 21)])
    df_auc_pr = pd.DataFrame(columns=['Dataset'] + [f'{(i * 0.05):.2f}' for i in range(0, 21)])
    df_time = pd.DataFrame(columns=['Dataset'] + [f'{(i * 0.05):2f}' for i in range(0, 21)])

    for dataset in all_datasets:
        data, decision_column = read_dataset(dataset=dataset)
 
        # Sigma is set from 0 to 1 with increment of 0.05
        sigma = 0.0

        time_history = []
        auc_roc_history = []
        auc_pr_history = []

        while sigma <= 1.0:
            start_time = time.time()
            print(f"--- Starting GBFRD-RBF ({dataset} dataset) with sigma = {sigma} ---")

            gbfrd_rbf = GBFRD_RBF(sigma=sigma)
            out_factors = gbfrd_rbf.calculate_outlier_factor(data)
            # print(f"Sigma: {sigma}, Outlier Factors: {out_factors}")

            # Calculate execution time
            elapsed_time = time.time() - start_time
            print("--- Completed: %s seconds ---" % (elapsed_time))
            time_history += [round(elapsed_time, 4)]
            # print(out_factors)  

            # Calculate detection rate
            detection_rate = np.sum(out_factors) / len(out_factors)
            print(f"Detection Rate: {detection_rate}")

            # Calculate AUC for GBFRD
            auc_roc_gbfrd_rbf = roc_auc_score(decision_column, out_factors)

            precision, recall, _ = precision_recall_curve(decision_column, out_factors)
            auc_pr_gbfrd_rbf = auc(recall, precision)
            print(f"GBFRD-RBF AUC-ROC: {auc_roc_gbfrd_rbf}")
            print(f"GBFRD-RBF AUC-PR: {auc_pr_gbfrd_rbf}")
            auc_roc_history += [round(auc_roc_gbfrd_rbf, 4)]
            auc_pr_history += [round(auc_pr_gbfrd_rbf, 4)]

            plot_auc_roc(decision_column, out_factors, "GBFRD_RBF", dataset, sigma)
            plot_auc_pr(decision_column, out_factors, "GBFRD_RBF", dataset, sigma)

            sigma = round(sigma + 0.05, 2)

        df_auc_roc.loc[len(df_auc_roc)] = [dataset] + auc_roc_history
        df_auc_pr.loc[len(df_auc_pr)] = [dataset] + auc_pr_history
        df_time.loc[len(df_time)] = [dataset] + time_history

        df_auc_roc.to_excel('Output/ROC_GBFRD_RBF.xlsx', index=False)
        df_auc_pr.to_excel('Output/PR_GBFRD_RBF.xlsx', index=False)
        df_time.to_excel('Output/ET_GBFRD_RBF.xlsx', index=False)



    # FREAD 

    df_auc_roc = pd.DataFrame(columns=['Dataset'] + [f'{(i * 0.05):.2f}' for i in range(0, 21)])
    df_auc_pr = pd.DataFrame(columns=['Dataset'] + [f'{(i * 0.05):.2f}' for i in range(0, 21)])
    df_time = pd.DataFrame(columns=['Dataset'] + [f'{(i * 0.05):2f}' for i in range(0, 21)])

    for dataset in all_datasets:
        data, decision_column = read_dataset(dataset=dataset)

        delta = 0.0

        time_history = []
        auc_roc_history = []
        auc_pr_history = []

        while delta <= 1.0:
            start_time = time.time()
            print(f"--- Starting FREAD ({dataset} dataset) with delta = {delta} ---")

            fread = FREAD(delta=delta)
            out_factors = fread.calculate_outlier_factor(data)
        
            # Calculate execution time
            elapsed_time = time.time() - start_time
            time_history += [round(elapsed_time, 4)]
            print("--- Completed: %s seconds ---" % (elapsed_time))
            # print(out_factors)  

            # Calculate detection rate
            detection_rate = np.sum(out_factors) / len(out_factors)
            print(f"Detection Rate: {detection_rate}")

            # Calculate AUC for FREAD
            auc_fread = roc_auc_score(decision_column, out_factors)
            precision, recall, _ = precision_recall_curve(decision_column, out_factors)
            auc_pr_fread = auc(recall, precision)

            # Open the file in append mode
            with open('log.txt', 'a') as file:
                file.write(f"FREAD ({dataset} dataset) with delta = {delta}" + "\n")
                file.write(f"FREAD AUC-ROC: {auc_fread}" + "\n")
                file.write(f"FREAD AUC-PR: {auc_pr_fread}" + "\n")
                file.write(f"FREAD Execution Time: {elapsed_time}" + "\n")


            print(f"FREAD AUC-ROC: {auc_fread}")
            print(f"FREAD AUC-PR: {auc_pr_fread}")
            auc_roc_history += [round(auc_fread, 4)]
            auc_pr_history += [round(auc_pr_fread, 4)]

            plot_auc_roc(decision_column, out_factors, "FREAD", dataset, delta)
            plot_auc_pr(decision_column, out_factors, "FREAD", dataset, delta)

            delta = round(delta + 0.05, 2)

        df_auc_roc.loc[len(df_auc_roc)] = [dataset] + auc_roc_history
        df_auc_pr.loc[len(df_auc_pr)] = [dataset] + auc_pr_history
        df_time.loc[len(df_time)] = [dataset] + time_history

        df_auc_roc.to_excel(f'Output/FREAD_ROC_{dataset}.xlsx', index=False)
        df_auc_pr.to_excel(f'Output/FREAD_PR_{dataset}.xlsx', index=False)
        df_time.to_excel(f'Output/FREAD_ET_{dataset}.xlsx', index=False)


        