import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils_nans1 import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.tools.tools import add_constant



def check_independence_of_observations(data):
    """
    Check the independence of observations assumption.

    Args:
    - data: DataFrame or array-like containing the dataset.

    Returns:
    - independence_satisfied: Boolean indicating if the independence of observations assumption is satisfied.
    """
    # Perform any necessary checks here
    # For example, check for autocorrelation in time series data
    # If the data is randomly sampled, this assumption is often satisfied by default
    independence_satisfied = True  # Placeholder, change based on checks
    return independence_satisfied

def check_random_sampling(data):
    """
    Check the random sampling assumption.

    Args:
    - data: DataFrame or array-like containing the dataset.

    Returns:
    - random_sampling_satisfied: Boolean indicating if the random sampling assumption is satisfied.
    """
    # Perform any necessary checks here
    # For example, check if the data was randomly sampled
    # If using cross-validation or a random split, this assumption is often satisfied by default
    random_sampling_satisfied = True  # Placeholder, change based on checks
    return random_sampling_satisfied

def check_variety_of_features(data):
    """
    Check the variety of features assumption.

    Args:
    - data: DataFrame or array-like containing the dataset.

    Returns:
    - variety_of_features_satisfied: Boolean indicating if the variety of features assumption is satisfied.
    """
    # Perform any necessary checks here
    # For example, check if there is sufficient diversity among features
    # If the dataset contains diverse features, this assumption is often satisfied
    variety_of_features_satisfied = True  # Placeholder, change based on checks
    return variety_of_features_satisfied

def check_feature_importance(data, model):
    """
    Check the feature importance assumption.

    Args:
    - data: DataFrame or array-like containing the dataset.
    - model: Trained Random Forest model.

    Returns:
    - feature_importance_satisfied: Boolean indicating if the feature importance assumption is satisfied.
    """
    # Perform any necessary checks here
    # For example, check if feature importances are significantly different
    # If important features contribute significantly to the model, this assumption is often satisfied
    feature_importance_satisfied = True  # Placeholder, change based on checks
    return feature_importance_satisfied

def check_random_forest_assumptions(data, model):
    """
    Check all assumptions for Random Forest model.

    Args:
    - data: DataFrame or array-like containing the dataset.
    - model: Trained Random Forest model.

    Returns:
    - assumptions_satisfied: Dictionary indicating if assumptions are satisfied.
    """
    assumptions_satisfied = {}
    assumptions_satisfied['Independence of Observations'] = check_independence_of_observations(data)
    assumptions_satisfied['Random Sampling'] = check_random_sampling(data)
    assumptions_satisfied['Variety of Features'] = check_variety_of_features(data)
    assumptions_satisfied['Feature Importance'] = check_feature_importance(data, model)
    return assumptions_satisfied
