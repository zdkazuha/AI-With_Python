import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

df = pd.read_csv('./assets/internship_candidates_cefr_final.csv')

X = df[['Experience', 'Grade', 'EnglishLevel', 'Age', 'EntryTestScore']]
y = df['Accepted']

categorical_features = ['EnglishLevel']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' 
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test) 

plt.scatter(X_test['EnglishLevel'], X_test['EntryTestScore'], c=y_pred, cmap='RdYlGn', edgecolor='k', s=100)
plt.title('Logistic Regression Predictions')
plt.xlabel('English Level')
plt.ylabel('Entry Test Score')    
plt.colorbar(label='Predicted Class')
plt.show()

new_data = pd.DataFrame({
    'Experience': [3, 4],
    'Grade': [8, 11],
    'EnglishLevel': ['Elementary', 'Intermediate'],
    'Age': [22, 29],
    'EntryTestScore': [634, 795]
})

predictions = model.predict(new_data)
print("Predictions for new data:", predictions)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))