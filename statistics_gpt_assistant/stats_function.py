import pandas as pd
from scipy.stats import shapiro, levene
import numpy as np

def check_normality(data, value_col):
    stat, p_value = shapiro(data[value_col])
    normal = p_value > 0.05  # Assuming alpha = 0.05
    return {
        'test' : 'saphiro',
        'normal' : normal,
        'p_value' : p_value,
        'text' : f"The Shapiro-Wilk test statistic return p-value: {p_value}" 
    }

def check_normality_of_groups(data, group_col, value_col):
    groups_normality_result = []
    for group in data[group_col].unique():
        normality_result = check_normality(data=data[data[group_col]==group], value_col=value_col)
        groups_normality_result.append({
            'test' : 'saphiro',
            'group' : group,
            'normality' : str(normality_result['normal']),
            'p-value' : str(normality_result['p_value']),
            'text' : f"The Shapiro-Wilk test statistic for group {group} return p-value: {str(normality_result['p_value'])}"
        })
    return normality_result

def check_homogeneity_of_variances(data, group_col, value_col):
    groups = data[group_col].unique()
    group_data = [data[data[group_col] == group][value_col] for group in groups]
    stat, p_value = levene(*group_data)
    equal_variances = p_value > 0.05  # Assuming alpha = 0.05
    return {
        'test' : 'levene',
        'equal_variance' : str(equal_variances),
        'p-value' : str(p_value),
        'text' : f"Levene's test statistic return p-value: {str(p_value)}"
    }