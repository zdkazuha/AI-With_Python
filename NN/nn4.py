import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import keras
from keras import layers

data = pd.read_csv("./assets/train.csv")
data_test = pd.read_csv("./assets/test.csv")

df = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = pd.get_dummies(df, columns=['Embarked'])

df['Age'] = df['Age'].fillna(df['Age'].median())

X = df.drop('Survived', axis=1)
y = df['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = keras.Sequential([
    layers.Input(shape=(9,)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam", 
    loss="binary_crossentropy", 
    metrics=["accuracy"]      
)
model.fit(X_train, y_train, epochs=50, batch_size=32)

model.evaluate(X_test, y_test)

sample = np.array([[1, 0, 26, 1, 0, 31.2833, 0, 0, 0]])
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled, verbose=0) 
probability = prediction[0][0]

status = "Вижив(Ла)" if probability > 0.5 else "Загинув(Ла)"

print("-" * 50)
print(f"Прогноз:")
print(f"Статус: {status}")