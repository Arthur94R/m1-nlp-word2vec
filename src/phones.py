import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Mode non-interactif (pas d'affichage)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

# ===========================
# ÉTAPE 1 : CHARGEMENT
# ===========================

print("=== CHARGEMENT DU DATASET PHONES ===\n")

DATA_PATH = os.path.join(DATA_DIR, 'Cell_Phones_and_Accessories_5.json')

df = pd.read_json(DATA_PATH, lines=True)

print(f"\nShape : {df.shape}")
print(f"\nColonnes : {df.columns.tolist()}")
print(f"\nPremières lignes :")
print(df.head())
print(f"\nInfo :")
print(df.info())
print(f"\nValeurs manquantes :")
print(df.isnull().sum())

# ===========================
# ÉTAPE 2 : EXPLORATION
# ===========================

print("\n=== ÉTAPE 2 : EXPLORATION ===\n")

# Variable cible : overall (note de 1 à 5)
print("=== Distribution de 'overall' (variable cible) ===")
print(df['overall'].value_counts().sort_index())

plt.figure(figsize=(8, 5))
df['overall'].value_counts().sort_index().plot(kind='bar', edgecolor='black', color='steelblue')
plt.title('Distribution des notes (overall)', fontweight='bold')
plt.xlabel('Note')
plt.ylabel('Nombre de reviews')
plt.xticks(rotation=0)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('../results/overall_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyse du texte
print("\n=== Analyse de reviewText ===")
df['text_length'] = df['reviewText'].apply(len)
print(f"Longueur moyenne : {df['text_length'].mean():.2f} caractères")
print(f"Longueur médiane : {df['text_length'].median():.2f} caractères")
print(f"Min : {df['text_length'].min()}")
print(f"Max : {df['text_length'].max()}")

# Exemples de reviews
print("\n=== EXEMPLES DE REVIEWS ===")
for i in range(3):
    print(f"\nReview {i+1} :")
    print(f"  Note    : {df['overall'].iloc[i]}/5")
    print(f"  Résumé  : {df['summary'].iloc[i]}")
    print(f"  Texte   : {df['reviewText'].iloc[i][:150]}...")

# ===========================
# ÉTAPE 3 : PREPROCESSING
# ===========================

print("\n=== ÉTAPE 3 : PREPROCESSING ===\n")

stop_words = set(stopwords.words('english'))

def preprocess(text):
    if pd.isna(text):
        return []
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return tokens

print("Preprocessing en cours...")
df['tokens'] = df['reviewText'].apply(preprocess)
df['num_tokens'] = df['tokens'].apply(len)
print(" Preprocessing terminé !\n")

# Stats
print("=== STATISTIQUES TOKENS ===")
print(f"Moyenne  : {df['num_tokens'].mean():.2f}")
print(f"Médiane  : {df['num_tokens'].median():.2f}")
print(f"Min      : {df['num_tokens'].min()}")
print(f"Max      : {df['num_tokens'].max()}")

# Vocabulaire
all_tokens = [token for tokens_list in df['tokens'] for token in tokens_list]
token_counts = Counter(all_tokens)
print(f"\nNombre total de tokens : {len(all_tokens):,}")
print(f"Taille du vocabulaire  : {len(token_counts):,}")

# Top 20
print("\n=== TOP 20 TOKENS ===")
for token, count in token_counts.most_common(20):
    print(f"  {token:20s} : {count:7,d}")

# ===========================
# ÉTAPE 4 : WORD2VEC
# ===========================

print("\n=== ÉTAPE 4 : WORD2VEC ===\n")

sentences = df['tokens'].tolist()

model_w2v = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=1,
    epochs=10
)

print(f" Modèle entraîné !")
print(f"Taille du vocabulaire (min_count=5) : {len(model_w2v.wv):,}\n")

# Mots similaires
test_words = ['phone', 'battery', 'screen', 'quality', 'case']
print("=== MOTS SIMILAIRES ===")
for word in test_words:
    if word in model_w2v.wv:
        similar = model_w2v.wv.most_similar(word, topn=5)
        print(f"\n'{word}' est proche de :")
        for sim_word, score in similar:
            print(f"  {sim_word:15s} : {score:.3f}")

model_w2v.save(os.path.join(DATA_DIR, 'word2vec_phones.bin'))
print("\n Modèle sauvegardé : data/word2vec_phones.bin")