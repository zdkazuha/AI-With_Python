import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
 
df = pd.read_csv("./assets/fuel_consumption_vs_speed.csv")
 
X = df[['speed_kmh']] 
y = df['fuel_consumption_l_per_100km']

degree = 3
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
 
model.fit(X, y)

X_test = pd.DataFrame({'speed_kmh': [35, 95, 140]})
y_pred = model.predict(X_test)

print("Прогноз витрати пального:")
for speed, fuel in zip(X_test['speed_kmh'], y_pred):
    print(f"Швидкість {speed} км/год -> {fuel:.2f} л/100км")

plt.figure(figsize=(10, 6))

plt.scatter(X, y, color='blue', alpha=0.5)

X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range_pred = model.predict(pd.DataFrame(X_range, columns=['speed_kmh']))

plt.plot(X_range, y_range_pred, color='red', linewidth=2)

plt.xlabel('Швидкість (км/год)')
plt.ylabel('Витрата (л/100км)')
plt.title('Аналіз витрати пального')
plt.grid(True)
plt.show()

all_pred = model.predict(X)

mae = mean_absolute_error(y, all_pred)
mse = mean_squared_error(y, all_pred)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}") 