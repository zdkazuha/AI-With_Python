import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("./assets/fuel_consumption_vs_speed.csv")

X = df[['speed_kmh']] 
y = df['fuel_consumption_l_per_100km']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

new_speeds = pd.DataFrame({'speed_kmh': [35, 95, 140]})
predicted_consumption = model.predict(new_speeds)

y_pred = model.predict(X_test)

print("Прогноз витрати пального:")
for speed, fuel in zip(new_speeds['speed_kmh'], predicted_consumption):
    print(f"Швидкість {speed} км/год -> {fuel:.2f} л/100км")

print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.4f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.4f}")

plt.figure(figsize=(10, 6))

plt.scatter(X_test, y_test, color='blue', alpha=0.7)
plt.scatter(new_speeds, predicted_consumption, color='green', s=100, zorder=5)

X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range_pred = model.predict(pd.DataFrame(X_range, columns=['speed_kmh']))
plt.plot(X_range, y_range_pred, color='red', linestyle='-',)

plt.xlabel("Швидкість (км/год)")
plt.ylabel("Витрата пального (л/100км)")
plt.title("Залежність витрати пального від швидкості")
plt.grid(True)
plt.show()