import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

import keras
from keras import layers

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

data = {
    'speed_kmh': [35, 95, 140, 50, 110, 80],
    'travel_time_hours': [1.0, 2.0, 1.5, 1.2, 1.8, 1.1],
    'engine_type': ['petrol', 'diesel', 'petrol', 'diesel', 'petrol', 'diesel'],
    'fuel_consumption_l_per_100km': [8.5, 5.2, 10.1, 5.5, 9.2, 5.8]
}

df = pd.DataFrame(data)

X = df[['speed_kmh', 'travel_time_hours', 'engine_type']]
y = df['fuel_consumption_l_per_100km']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['engine_type'])
    ],
    remainder='passthrough'
)

X_transformed = preprocessor.fit_transform(X)

model_nn = keras.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(8, activation="relu"),
    layers.Dense(1)
])

model_nn.compile(optimizer="adam", loss="mse", metrics=["mae"])
model_nn.fit(X_transformed, y, epochs=200, verbose="silent")

model_poly = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly', PolynomialFeatures(degree=2)),
    ('regressor', LinearRegression())
])

model_poly.fit(X, y)

test_data = pd.DataFrame({
    'speed_kmh': [35, 95, 140],
    'travel_time_hours': [1.0, 2.0, 1.5],
    'engine_type': ['petrol', 'diesel', 'petrol']
})

test_transformed = preprocessor.transform(test_data)
nn_pred = model_nn.predict(test_transformed)
poly_pred = model_poly.predict(test_data)

print(f"{'Швидкість':<12} | {'Нейромережа':<15} | {'Поліном':<15}")
print("-" * 50)
for i in range(len(test_data)):
    print(f"{test_data.iloc[i]['speed_kmh']:<12} | {nn_pred[i][0]:<15.2f} | {poly_pred[i]:<15.2f}")