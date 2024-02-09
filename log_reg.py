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






def independence_of_errors_assumptio(model, features, labels, plot=False):
  
   
    if hasattr(model, 'predict'):  # Provera da li model ima metodu 'predict'
        predicted_values = model.predict(features)
    else:
        raise ValueError("Model does not have a 'predict' method.")

    # Izračunavanje reziduala
    residuals = labels - predicted_values

    if plot:
        plt.scatter(predicted_values, residuals)
        plt.axhline(y=0, color='darkorange', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Scatterplot of Predicted Values vs. Residuals')
        plt.show()

    from statsmodels.stats.stattools import durbin_watson
    dw_value = durbin_watson(residuals)

    autocorrelation = None
    if dw_value < 1.5:
        autocorrelation = 'positive'
    elif dw_value > 2:
        autocorrelation = 'negative'
    else:
        autocorrelation = None
    return autocorrelation, dw_value

def linearity_assumption(model, features, labels):
    '''
    Linearity assumption: assumes that the relationship between the features and the log odds of the response variable is linear.
    Testing linearity by checking the linearity of residuals with predicted values.

    Returns:
    - linearity_satisfied: A boolean indicating if the linearity assumption is satisfied.
    '''
    # Calculate residuals
    predicted_values = model.predict(features)
    residuals = labels - predicted_values
    
    # Check linearity by examining the residuals
    linearity_satisfied = np.allclose(residuals, 0, atol=1e-5)
    
    return linearity_satisfied



def absence_of_multicollinearity(features, threshold=5.0):
    """
    Check for absence of multicollinearity using Variance Inflation Factor (VIF).

    Args:
    - features: DataFrame containing the features.
    - threshold: Threshold value for identifying multicollinearity. Default is 5.0.

    Returns:
    - multicollinearity_satisfied: A boolean indicating if absence of multicollinearity is satisfied.
    - vif_values: Series containing VIF values for each feature.
    """
    # Convert non-numeric columns to numeric
    non_numeric_columns = features.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in non_numeric_columns:
        features[col] = LabelEncoder().fit_transform(features[col])

    # Calculate VIF for each feature
    vif_values = pd.Series(
        [variance_inflation_factor(features.values, i) for i in range(features.shape[1])],
        index=features.columns
    )

    # Check if any VIF value exceeds the threshold
    multicollinearity_satisfied = not (vif_values > threshold).any()

    return multicollinearity_satisfied, vif_values