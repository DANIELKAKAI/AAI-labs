import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import joblib


df = pd.read_csv('../data.csv')

X = df[['Continent', 'Subject Descriptor']]
y = df[[str(i) for i in range(1980,2026)]] 

y = y.apply(pd.to_numeric, errors='coerce')

imputer = SimpleImputer(strategy='mean')
y_imputed = imputer.fit_transform(y)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), ['Continent', 'Subject Descriptor'])], remainder='passthrough')
X_encoded = ct.fit_transform(X)

model = LinearRegression()
model.fit(X_encoded, y_imputed)

filename = 'gdp_model.sav'
joblib.dump(model, filename)