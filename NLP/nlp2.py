from textblob import TextBlob
import pandas as pd

df = pd.read_csv("./assets/reviews_list.csv")

reviews = df['Review Text']

neutral, good, bad = [], [], []

for text in reviews:
    
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    if sentiment >= 0.3:
        good.append(text)
    elif sentiment <     -0.3:
        bad.append(text)
    else:
        neutral.append(text)

print(f"[Count Neutral] --> {len(neutral)}")
print(f"[Count Good] -----> {len(good)}")
print(f"[Count Bad] ------> {len(bad)}")