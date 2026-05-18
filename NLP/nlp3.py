import string
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.decomposition import PCA

# 1

text = """
The sound quality of these earbuds is actually very good and impressive.
I had a really bad experience with the customer support of this brand.
This product offers amazing value and the build quality is decent.
It is a good pair of headphones but the Bluetooth connection is bad.
The noise cancellation is complete trash and the service is very bad.
They provide excellent quality and good performance for a low price.
The audio quality is crystal clear and I am having a good time using it.
The microphone is bad and nobody can hear my voice during calls.
AeroSound Pro delivers premium quality which is surprisingly good.
Do not buy this item because the battery life is incredibly bad.
They managed to combine good sound and solid quality in one device.
The design is nice but the overall performance is just bad.
We deserve better quality for this money, this is a bad joke.
Everything about this product is good, from texture to sound quality.
The cheap plastic makes the whole experience feel bad and annoying.
"""

translator = str.maketrans("", "", string.punctuation)
data = []

sentences = sent_tokenize(text)

for sentence in sentences:
    clean_sentence = sentence.lower().translate(translator)
    tokens = word_tokenize(clean_sentence)
    if tokens:
        data.append(tokens)

num_sentences = len(data)
total_words = sum(len(sentence) for sentence in data)
avg_length = total_words / max(num_sentences, 1)

print(f"Кількість речень: {num_sentences}")
print(f"Середня довжина речення: {avg_length:.2f}\n")

# 2

model = Word2Vec(data, vector_size=100, window=5, min_count=1, sg=1)

vector = model.wv["good"]
print("Вектор для слова 'good':\n", vector, "\n")

# 3

target_words = ["good", "bad", "quality"]

print(f"{'Цільове слово':<15} | {'Близькі слова':<25} | {'Косинусна схожість'}")
print("-" * 65)

for word in target_words:
    if word in model.wv:
        similar_results = model.wv.most_similar(word, topn=3)

        words_str = ", ".join([res[0] for res in similar_results])
        scores_str = ", ".join([f"{res[1]:.2f}" for res in similar_results])

        print(f"{word:<15} | {words_str:<25} | {scores_str}")
print()

# 4

def visualize(model, target_word):
    if target_word not in model.wv:
        return

    similar_results = model.wv.most_similar(target_word, topn=5)
    words = [target_word] + [res[0] for res in similar_results]
    vectors = [model.wv[w] for w in words]

    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)

    plt.figure(figsize=(7, 5))

    for i, word in enumerate(words):
        color = "blue" if i == 0 else "#1fb44c"
        
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1], color=color, s=100)
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), xytext=(5, 5), textcoords="offset points")

    plt.title(f"Схожі слова до '{target_word}'")
    plt.grid(True)
    plt.show()


visualize(model, "good")
visualize(model, "bad")
visualize(model, "quality")