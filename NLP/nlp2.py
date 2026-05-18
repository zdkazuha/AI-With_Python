from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./assets/reviews_list.csv")

reviews = df['Review Text']

neutral, good, bad = [], [], []

for text in reviews:
    
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    if sentiment >= 0.3:
        good.append(text)
    elif sentiment < -0.3:
        bad.append(text)
    else:
        neutral.append(text)

categories = ['Нейтральні', 'Позитивні', 'Негативні']
counts = [len(neutral), len(good), len(bad)]
colors = ['#afb8c1', '#2da44e', '#cf222e']  

plt.figure(figsize=(8, 5))
bars = plt.bar(categories, counts, color=colors, edgecolor='black', alpha=0.8)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontsize=11)

plt.title('Розподіл відгуків за тональністю (Sentiment Analysis)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Категорія відгуку', fontsize=12, labelpad=10)
plt.ylabel('Кількість відгуків', fontsize=12, labelpad=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()