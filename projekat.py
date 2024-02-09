import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils_nans1 import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import *
from sklearn.decomposition import PCA
from statsmodels.tools.tools import add_constant
from random_forest import *
from log_reg import *


df = pd.read_csv('./booking.csv')

# One hot encoding za ove kategorije
categorical_columns = ["room type", "type of meal"]
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
df = pd.get_dummies(df, columns=['market segment type'], prefix='market_segment')

#booking status - numericke vrednosti
df['booking status'] = df['booking status'].map({'Canceled': 1, 'Not_Canceled': 0})


df['date of reservation'] = pd.to_datetime(df['date of reservation'], errors='coerce')
df['month_of_reservation'] = df['date of reservation'].dt.month
df['year_of_reservation'] = df['date of reservation'].dt.year
df = df.dropna(subset=['month_of_reservation', 'year_of_reservation'])
df['month_of_reservation'] = df['month_of_reservation'].astype(int)
df['year_of_reservation'] = df['year_of_reservation'].astype(int)

df = df.drop(columns=['Booking_ID'])

#print(perfect_collinearity_assumption(df))
#outliers, outlier_indices = check_outliers_iqr(df[['average price']])
#cleaned_df = df.drop(outlier_indices)
########## korelacija
# Generisanje korelacione matrice
correlation_matrix = df.corr()
np.fill_diagonal(correlation_matrix.values, 0)

# Pronalaženje kolona sa visokom korelacijom
high_correlation_features = correlation_matrix[abs(correlation_matrix) > 0.7]
high_correlation_features = high_correlation_features.stack().index.tolist()

# Izbor samo jedne kolone za izbacivanje
column_to_drop = high_correlation_features[0][0]  # Uzima se prva kolona iz prvog para sa visokom korelacijom

# Provera da li kolona postoji pre nego što se izvrši drop
if column_to_drop in df.columns:
    df = df.drop(columns=[column_to_drop])

#print("high")
#print(column_to_drop)
########
# Izračunavanje Pearsonovog koeficijenta korelacije između atributa i ciljne promenljive
correlation = df.corr()['booking status'].abs()

# Određivanje praga za korelaciju ispod kojeg ćemo izbaciti atribute
threshold = 0.05 # Primer praga, možete prilagoditi prema potrebi

# Izdvajanje atributa čija je korelacija sa ciljnom promenljivom ispod praga
low_correlation_features = correlation[correlation < threshold].index.tolist()

#print("low")
#print(low_correlation_features)

# Izbacivanje nisko korelisanih atributa iz DataFrame-a
df = df.drop(columns=low_correlation_features)


x = df.drop(columns=['booking status'])
y = df['booking status']
x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=42)




# Normalizacija podataka
scaler = StandardScaler()
x_train_normalized = scaler.fit_transform(x_train)
x_val_normalized = scaler.transform(x_val)

# Primena PCA na trening podacima
pca_model = PCA(n_components=7, random_state=42)
principal_components_train = pca_model.fit_transform(x_train_normalized)

# Transformacija validacionih podataka koristeći isti PCA model
principal_components_val = pca_model.transform(x_val_normalized)
log_reg = LogisticRegression(max_iter=1000)

# Obuka modela logističke regresije na trening podacima
log_reg.fit(x_train_normalized, y_train)

# Predikcija rezultata na test skupu
y_pred = log_reg.predict(x_val_normalized)

# Evaluacija performansi modela
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)

# Izveštaj klasifikacije
print("\nClassification report:")
print(classification_report(y_val, y_pred))

# Poziv funkcije independence_of_errors_assumption
autocorrelation, dw_value = independence_of_errors_assumptio(log_reg, x_val, y_val)

# Ispis rezultata
print("Autocorrelation:", autocorrelation)
print("Durbin-Watson value:", dw_value) # Vrednost Durbin-Watson statistike koja je blizu 2 (u ovom slučaju 1.94) sugeriše da nema ozbiljne autocorelacione strukture u rezidualima.


# Provera pretpostavki
linearity_satisfied = linearity_assumption(log_reg, x_train, y_train)
print("Linearity assumption satisfied:", linearity_satisfied)

# Poziv funkcije za odsustvo multicollinearity
multicollinearity_satisfied, vif_values = absence_of_multicollinearity(x_train)
print("Absence of multicollinearity satisfied:", multicollinearity_satisfied)

######### poly

# Kreiranje polinomijalnih članova
poly = PolynomialFeatures(degree=3)
x_train_poly = poly.fit_transform(x_train)
x_val_poly = poly.transform(x_val)

# Inicijalizacija i obuka modela
log_reg_poly = LogisticRegression()
log_reg_poly.fit(x_train_poly, y_train)

# Predikcija rezultata na validacionom skupu
y_pred_poly = log_reg_poly.predict(x_val_poly)

# Evaluacija performansi modela
accuracy_poly = accuracy_score(y_val, y_pred_poly)
print("Accuracy(poly):", accuracy_poly)

print("\nClassification_report(poly):")
print(classification_report(y_val, y_pred_poly))

# Pretvaranje NumPy ndarray u DataFrame
x_train_poly_df = pd.DataFrame(x_train_poly, columns=poly.get_feature_names_out(x_train.columns))

autocorrelation_poly, dw_value_poly = independence_of_errors_assumptio(log_reg_poly, x_val_poly, y_val)

# Ispis rezultata
print("Autocorrelation:", autocorrelation_poly)
print("Durbin-Watson value:", dw_value_poly) 

linearity_satisfied_poly = linearity_assumption(log_reg_poly, x_train_poly, y_train)
print("Linearity assumption satisfied:", linearity_satisfied_poly)




########


########
"""
# Inicijalizacija MinMaxScaler-a
scaler = MinMaxScaler()

# Normalizacija podataka
x_train_normalized = scaler.fit_transform(x_train)
x_val_normalized = scaler.transform(x_val)



# Inicijalizacija i obuka modela logističke regresije
log_reg = LogisticRegression(max_iter=1000)

# Obuka modela na trening podacima
log_reg.fit(x_train_normalized, y_train)

# Predikcija rezultata na validacionom skupu
y_pred_norm = log_reg.predict(x_val_normalized)

# Evaluacija performansi modela
accuracy = accuracy_score(y_val, y_pred_norm)
print("Accuracy(normalized):", accuracy)

# Izveštaj klasifikacije
print("\nClassification_report(normalized):")
print(classification_report(y_val, y_pred_norm))

linearity_logg = linearity_assumption(log_reg, x_train_normalized, y_train)
print("Linearity N", linearity_logg)"""


###### RANDOM FOREST

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

random_forest.fit(x_train, y_train)

y_pred_rf = random_forest.predict(x_val)

# Evaluacija performansi modela
accuracy_rf = accuracy_score(y_val, y_pred_rf)
print("Accuracy(random forest):", accuracy_rf)

# Izveštaj klasifikacije Random Forest modela
print("\nClassification report(random forest):")
print(classification_report(y_val, y_pred_rf))


assumptions_satisfied = check_random_forest_assumptions(x_train, random_forest)
for assumption, satisfied in assumptions_satisfied.items():
     print(f"{assumption}: {'Yes' if satisfied else 'No'}")
     
#####