import glob
import os
import pandas as pd
import numpy as np
import openpyxl
from scipy.stats import ttest_rel, wilcoxon, shapiro, levene, f_oneway

# Define the path to the parent folder
RESULTS_PATH = os.path.join("results", "MLP_tree", "hidden_size_256")

# Define the paths to the two folders to compare
folders_regular = [
    "s_MLP_tree_data_insilico_seed_56_2024-04-04_PH_",
    "s_MLP_tree_data_insilico_seed_8_2024-05-30_PH_",
    "s_MLP_tree_data_insilico_seed_9_2024-05-30_PH_",
    "s_MLP_tree_data_insilico_seed_25_2024-05-30_PH_",
    "s_MLP_tree_data_insilico_seed_44_2024-05-30_PH_",
    "s_MLP_tree_data_insilico_seed_101_2024-05-30_PH_",
    "s_MLP_tree_data_insilico_seed_78_2024-05-30_PH_",
    "s_MLP_tree_data_insilico_seed_61_2024-05-30_PH_",
    "s_MLP_tree_data_insilico_seed_2042_2024-05-30_PH_",
    "s_MLP_tree_data_insilico_seed_732_2024-05-30_PH_"
]
folders_causal = [
    "s_MLP_tree_t_simglucose_data_insilico_seed_56_2024-04-04_PH_",
    "s_MLP_tree_t_simglucose_data_insilico_seed_8_2024-05-30_PH_",
    "s_MLP_tree_t_simglucose_data_insilico_seed_9_2024-05-30_PH_",
    "s_MLP_tree_t_simglucose_data_insilico_seed_25_2024-05-30_PH_",
    "s_MLP_tree_t_simglucose_data_insilico_seed_44_2024-05-30_PH_",
    "s_MLP_tree_t_simglucose_data_insilico_seed_101_2024-05-30_PH_",
    "s_MLP_tree_t_simglucose_data_insilico_seed_78_2024-05-30_PH_",
    "s_MLP_tree_t_simglucose_data_insilico_seed_61_2024-05-30_PH_",
    "s_MLP_tree_t_simglucose_data_insilico_seed_2042_2024-05-30_PH_",
    "s_MLP_tree_t_simglucose_data_insilico_seed_732_2024-05-30_PH_"
]

# PHS = ["30","45","60","120"]
PH = "60"

def extract_values(directory_paths):


    # Lists to store the values
    mse_values = []
    mae_values = []
    ega_values = []

    for path_seed in directory_paths:
        file = os.path.join(RESULTS_PATH,path_seed+PH, f"output.xlsx")

        # Read the xlsx file
        wb = openpyxl.load_workbook(file, data_only=True)
        sheet = wb.active
        
        # Read the value from cell mse
        if not sheet["B5"].value:
            raise ValueError("No MSE value")
        mse_values.append(float(sheet['B5'].value)/0.0001)
        
        # Read the value from cell mae
        if not sheet["C5"].value:
            raise ValueError("No MAE value")
        mae_values.append(float(sheet['C5'].value)/0.01)
        
        # Calculate the sum of D5 and E5 and append
        if not sheet["D5"].value or\
            not sheet["E5"].value :
            raise ValueError("No EGA value")
        d5_value = sheet['D5'].value
        e1_value = sheet['E5'].value
        ega_values.append((d5_value + e1_value)/30)

    # Convert lists to numpy arrays for statistical calculations
    mse_values = np.array(mse_values, dtype=float)
    mae_values = np.array(mae_values, dtype=float)
    ega_values = np.array(ega_values, dtype=float)

    return mse_values, mae_values, ega_values

def test_difference(
    values_regular,
    values_causal
):
    # Check for normality
    _, p_normality_model_mse_1 = shapiro(values_regular)
    _, p_normality_model_mse_2 = shapiro(values_causal)

    # Check for homogeneity of variances
    _, p_levene = levene(values_regular, values_causal)

    _, p_value = f_oneway(values_regular, values_causal)

    # If both tests above are True, you can use t-test
    # if p_normality_model_mse_1 > 0.05 and p_normality_model_mse_2 > 0.05 and p_levene > 0.05:
    #     # Perform the paired t-test
    #     print("t-test")
    #     _, p_value = ttest_rel(values_regular, values_causal)
    # else:
    #     print("wilcoxon")
    #     # Perform the Wilcoxon Signed-Rank Test
    #     _, p_value = wilcoxon(values_regular, values_causal)
    return p_value


df = pd.DataFrame(columns=['ph', 'mode', 'mean IIT', 'st IIT','mean S', 'st S', 'p-value', 'significant'])
alpha = 0.05

# Extract values from both directories

values_mse_regular, values_mae_regular, values_ega_regular = extract_values(folders_regular)
print(values_mse_regular)
values_mse_causal, values_mae_causal, values_ega_causal = extract_values(folders_causal)
print(values_mse_causal)

p_value_mse = test_difference(values_mse_regular, values_mse_causal)
mean_iit = np.mean(values_mse_causal)
std_iit = np.std(values_mse_causal)
mean_s = np.mean(values_mse_regular)
std_s = np.std(values_mse_regular)
new_row_mse = pd.DataFrame({'ph': PH, 'mode': 'MSE', 'mean IIT': mean_iit,'st IIT': std_iit,'mean S': mean_s,'st S': std_s,'p-value': p_value_mse, 'significant': p_value_mse < alpha}, index=[0])
df = pd.concat([df, new_row_mse], ignore_index=True)
p_value_mae = test_difference(values_mae_regular, values_mae_causal)
mean_iit = np.mean(values_mae_causal)
std_iit = np.std(values_mae_causal)
mean_s = np.mean(values_mae_regular)
std_s = np.std(values_mae_regular)
new_row_mae = pd.DataFrame({'ph': PH, 'mode': 'MAE', 'mean IIT': mean_iit,'st IIT': std_iit,'mean S': mean_s,'st S': std_s, 'p-value': p_value_mae, 'significant': p_value_mae < alpha}, index=[0])
df = pd.concat([df, new_row_mae], ignore_index=True)
mean_iit = np.mean(values_ega_causal)
std_iit = np.std(values_ega_causal)
mean_s = np.mean(values_ega_regular)
std_s = np.std(values_ega_regular)
new_row_ega = pd.DataFrame({'ph': PH, 'mode': 'EGA', 'mean IIT': mean_iit,'st IIT': std_iit,'mean S': mean_s,'st S': std_s, 'p-value': "-", 'significant': "-"}, index=[0])
df = pd.concat([df, new_row_ega], ignore_index=True)
print(df)
