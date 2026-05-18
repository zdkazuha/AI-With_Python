import pandas as pd
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# --- 1. ЗАВАНТАЖЕННЯ ТА ПІДГОТОВКА ДАНИХ ---
print("Завантаження датасету...")
# Завантажуємо датасет через Hugging Face
raw_data = load_dataset("dair-ai/emotion", trust_remote_code=True)

# Перетворюємо всі частини (train, test, validation) у один DataFrame для зручності
df_train = pd.DataFrame(raw_data['train'])
df_val = pd.DataFrame(raw_data['validation'])
df_test = pd.DataFrame(raw_data['test'])
df = pd.concat([df_train, df_val, df_test])

# Фільтруємо лише потрібні за завданням емоції:
# 0: sadness (сум), 1: joy (радість), 3: anger (злість)
target_emotions = [0, 1, 3]
df = df[df['label'].isin(target_emotions)]

# Перетворюємо мітки на послідовні: 0 -> 0, 1 -> 1, 3 -> 2
# Це важливо для коректної роботи вихідного шару Softmax
label_mapping = {0: 0, 1: 1, 3: 2}
df['label'] = df['label'].map(label_mapping)

texts = df['text'].values
labels = df['label'].values

# --- 2. ТОКЕНІЗАЦІЯ ТА ПАДДІНГ ---
max_words = 10000 
max_len = 50 

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len, padding='post')
y = np.array(labels)

# Розподіл на навчання та тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. ПОБУДОВА МОДЕЛІ RNN (LSTM) ---
model = Sequential([
    # 64-вимірний простір для кожного слова
    Embedding(max_words, 64, input_length=max_len),
    # LSTM краще за звичайну RNN, бо "пам'ятає" контекст довше
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation='relu'),
    Dropout(0.3),
    # 3 нейрони на виході (сум, радість, злість)
    Dense(3, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

# --- 4. НАВЧАННЯ ---
print("Початок навчання...")
model.fit(
    X_train, y_train, 
    epochs=5, 
    validation_data=(X_test, y_test), 
    batch_size=64
)

# --- 5. ФУНКЦІЯ ПРОГНОЗУВАННЯ ---
def predict_emotion(sentence):
    # Текст має пройти ту ж обробку, що і при навчанні
    seq = tokenizer.texts_to_sequences([sentence])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    prediction = model.predict(padded, verbose=0)
    
    classes = ['sadness', 'joy', 'anger']
    result_index = np.argmax(prediction)
    return classes[result_index]

# --- ТЕСТУВАННЯ ---
print("\n--- РЕЗУЛЬТАТИ ТЕСТУ ---")
test_phrases = [
    "I am so furious about this delay!",    # Очікуємо anger
    "This is the best day of my life",      # Очікуємо joy
    "I feel so lonely and empty inside"     # Очікуємо sadness
]

for phrase in test_phrases:
    print(f"Phrase: '{phrase}' -> Predicted: {predict_emotion(phrase)}")