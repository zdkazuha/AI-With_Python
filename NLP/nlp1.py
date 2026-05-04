import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd

# Завантаження необхідних мовних ресурсів
nltk.download("punkt")  # Для токенізації
nltk.download("punkt_tab")  # Для токенізації речень
nltk.download("stopwords")  # Стоп-слова
nltk.download("wordnet")  # Для лемматизації
nltk.download("omw-1.4")  # WordNet мовні дані

df = pd.read_csv("./assets/reviews_list.csv")

reviews = df['Review Text']

position = 0

data = {
    "Position": [],
    "Review Text": []
    }

for text in reviews:
    position += 1

    # --- 1. Токенізація ---
    sentences = sent_tokenize(text)

    words = word_tokenize(text)

    # --- 2. Стоп-слова ---
    stop_words = set(stopwords.words("english"))
    filtered_words = [
        word for word in words if word.lower() not in stop_words and word.isalpha()
    ]

    # --- 3. Стеммінг ---
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    # --- 4. Лемматизація ---
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in filtered_words]

    # --- 5. Збереження ---
    data["Position"].append(position)
    data["Review Text"].append(lemmatized_words)

# --- 6. Запис ---

df = pd.DataFrame(data)

df.to_csv('./assets/reviews_list_new.csv', index=False, encoding='utf-8-sig')