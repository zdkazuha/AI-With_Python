import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model_nn = keras.Sequential(
    [
        layers.Dense(16, activation="relu", input_shape=(1,)),  
        layers.Dense(4, activation="relu"),
        layers.Dense(1), 
    ]
)

model_nn.compile(optimizer="adam", loss="mse", metrics=["mae"])

df = pd.read_csv("./assets/fuel_consumption_vs_speed.csv")
X = df[['speed_kmh']]
y = df['fuel_consumption_l_per_100km']

model_nn.fit(X, y, epochs=200, verbose=0)

example = np.array([[35], [95], [140]])
pred = model_nn.predict(example)

degree = 3
model_poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model_poly.fit(X, y)

X_test = pd.DataFrame({'speed_kmh': [35, 95, 140]})
y_pred = model_poly.predict(X_test)

print("\n" + "="*60)
print(f"{'Швидкість':<12} | {'Нейромережа (NN)':<18} | {'Поліном (Poly)':<15}")
print("-" * 60)

for i in range(len(example)):
    speed = example[i][0]
    nn_val = pred[i][0]
    poly_val = y_pred[i]
    
    diff = abs(nn_val - poly_val)
    
    print(f"{speed:<12} | {nn_val:<18.4f} | {poly_val:<15.4f}")

print("-" * 60)
print(f"Висновок: На швидкості 35 км/год різниця складає {abs(pred[0][0] - y_pred[0]):.2f} л/100км")
print(f"Висновок: На швидкості 95 км/год різниця складає {abs(pred[1][0] - y_pred[1]):.2f} л/100км")
print(f"Висновок: На швидкості 140 км/год різниця складає {abs(pred[2][0] - y_pred[2]):.2f} л/100км")
print("="*60)


nn_train_pred = model_nn.predict(X)
poly_train_pred = model_poly.predict(X)

print(f"\nMAE Нейромережі: {metrics.mean_absolute_error(y, nn_train_pred):.4f}")
print(f"MAE Полінома:    {metrics.mean_absolute_error(y, poly_train_pred):.4f}")