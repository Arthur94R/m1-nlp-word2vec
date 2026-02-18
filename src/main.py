# Manipulation de données
import pandas as pd
import numpy as np

# Visualisation
import matplotlib
matplotlib.use('Agg')  # Mode non-interactif (pas d'affichage)
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing texte
import re
from collections import Counter

# NLTK pour le traitement du langage
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Word2Vec
from gensim.models import Word2Vec

# Machine Learning classique
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import StandardScaler

# Utilitaires
import warnings
warnings.filterwarnings('ignore')

# Configuration matplotlib
plt.style.use('default')
sns.set_palette("husl")


'''
# Télécharger les ressources NLTK (à faire une seule fois)
print("Téléchargement des ressources NLTK...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

print("✓ Imports terminés !")
'''

# ===========================
# ÉTAPE 1 : CHARGEMENT DES DONNÉES
# ===========================

print("=== DÉBUT DU TP : CHARGEMENT DES DONNÉES ===\n")

# Chargement du dataset principal
df = pd.read_csv('../data/movies_metadata.csv', low_memory=False)

print(f"Shape du dataset : {df.shape}")
print(f"Nombre de films : {df.shape[0]}")
print(f"Nombre de colonnes : {df.shape[1]}")

# Aperçu des colonnes
print("\n=== COLONNES DISPONIBLES ===")
print(df.columns.tolist())

# Premières lignes
print("\n=== PREMIÈRES LIGNES ===")
print(df.head())

# Informations sur les colonnes
print("\n=== INFO SUR LES COLONNES ===")
print(df.info())

# Valeurs manquantes
print("\n=== VALEURS MANQUANTES ===")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(missing)

# Focus sur la colonne 'overview' (descriptions)
print("\n=== ANALYSE DE LA COLONNE 'overview' ===")
print(f"Descriptions manquantes : {df['overview'].isnull().sum()}")
print(f"Descriptions présentes : {df['overview'].notna().sum()}")

# Exemples de descriptions
print("\n=== EXEMPLES DE DESCRIPTIONS ===")
for i in range(3):
    if pd.notna(df['overview'].iloc[i]):
        print(f"\n{i+1}. Titre : {df['title'].iloc[i]}")
        print(f"   Description : {df['overview'].iloc[i][:200]}...")


# ===========================
# ÉTAPE 2 : VARIABLES À PRÉDIRE
# ===========================

print("\n" + "="*60)
print("=== ÉTAPE 2 : VARIABLES INTÉRESSANTES À PRÉDIRE ===")
print("="*60 + "\n")

# Conversion des colonnes numériques (certaines sont en 'object')
numeric_cols = ['budget', 'popularity', 'vote_average', 'vote_count', 'revenue']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Variables cibles potentielles
target_candidates = ['vote_average', 'revenue', 'popularity', 'vote_count']

print("=== STATISTIQUES DES VARIABLES CIBLES ===\n")
for col in target_candidates:
    print(f"{col}:")
    print(f"  Min     : {df[col].min():.2f}")
    print(f"  Max     : {df[col].max():.2f}")
    print(f"  Moyenne : {df[col].mean():.2f}")
    print(f"  Médiane : {df[col].median():.2f}")
    print(f"  Écart-type     : {df[col].std():.2f}")
    print(f"  NaN     : {df[col].isnull().sum()}")
    print()

# Histogrammes des variables cibles
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, col in enumerate(target_candidates):
    axes[idx].hist(df[col].dropna(), bins=50, edgecolor='black', color='steelblue')
    axes[idx].set_title(f'Distribution de {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Fréquence')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/distributions_variables_cibles.png', dpi=300, bbox_inches='tight')
print("Graphique sauvegardé : distributions_variables_cibles.png")
plt.show()

# Remarques sur les distributions
print("\n=== REMARQUES SUR LES DISTRIBUTIONS ===")
print("""
1. vote_average : 
   - Distribution en cloche (gaussienne)
   - Centré autour de 6-7/10
   - La plupart des films ont une note moyenne

2. revenue :
   - Distribution très asymétrique (long tail)
   - Beaucoup de films avec revenue = 0 ou très faible
   - Quelques blockbusters avec revenus énormes

3. popularity :
   - Distribution log-normale
   - La majorité des films ont une faible popularité
   - Quelques films très populaires

4. vote_count :
   - Distribution très asymétrique aussi
   - Beaucoup de films avec peu de votes
   - Quelques films très votés
""")

# ===========================
# ÉTAPE 3 : PREPROCESSING DU TEXTE
# ===========================

print("\n" + "="*60)
print("=== ÉTAPE 3 : PREPROCESSING DES DESCRIPTIONS ===")
print("="*60 + "\n")

# Stop words en anglais
stop_words = set(stopwords.words('english'))
print(f"Nombre de stop words : {len(stop_words)}")
print(f"Exemples : {list(stop_words)[:15]}\n")

# Fonction de preprocessing
def preprocess(text):
    """
    1. Lowercase
    2. Garder uniquement lettres et espaces
    3. Tokenisation
    4. Filtrer stop words et mots courts (<=2)
    """
    if pd.isna(text):
        return []
    
    # Lowercase
    text = text.lower()
    
    # Garder uniquement lettres et espaces
    text = re.sub(r"[^a-zA-Z ]", "", text)
    
    # Tokenisation
    tokens = word_tokenize(text)
    
    # Filtrer stop words + mots courts
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    
    return tokens

# Créer un dataset nettoyé (sans les NaN dans overview)
df_clean = df[df['overview'].notna()].copy()
print(f"Films avec description : {len(df_clean)}")
print(f"Films supprimés (NaN) : {len(df) - len(df_clean)}\n")

# Appliquer le preprocessing
print("Preprocessing en cours...")
df_clean['tokens'] = df_clean['overview'].apply(preprocess)
print("Preprocessing terminé !\n")

# Exemples avant/après
print("=== EXEMPLES AVANT/APRÈS PREPROCESSING ===\n")
for i in range(3):
    print(f"Film {i+1} : {df_clean['title'].iloc[i]}")
    print(f"  Original : {df_clean['overview'].iloc[i][:120]}...")
    print(f"  Tokens   : {df_clean['tokens'].iloc[i][:15]}")
    print(f"  Nb tokens: {len(df_clean['tokens'].iloc[i])}")
    print()

# Statistiques sur les tokens
df_clean['num_tokens'] = df_clean['tokens'].apply(len)

print("=== STATISTIQUES TOKENS PAR DESCRIPTION ===")
print(f"Moyenne  : {df_clean['num_tokens'].mean():.2f}")
print(f"Médiane  : {df_clean['num_tokens'].median():.2f}")
print(f"Min      : {df_clean['num_tokens'].min()}")
print(f"Max      : {df_clean['num_tokens'].max()}")
print(f"Écart-type      : {df_clean['num_tokens'].std():.2f}")

# Histogramme nombre de tokens
plt.figure(figsize=(12, 5))
plt.hist(df_clean['num_tokens'], bins=50, edgecolor='black', color='green', alpha=0.7)
plt.title('Distribution du nombre de tokens par description (après preprocessing)', 
          fontsize=12, fontweight='bold')
plt.xlabel('Nombre de tokens')
plt.ylabel('Fréquence')
plt.grid(True, alpha=0.3)
plt.axvline(df_clean['num_tokens'].mean(), color='red', linestyle='--', 
            label=f'Moyenne = {df_clean["num_tokens"].mean():.1f}')
plt.legend()
plt.tight_layout()
plt.savefig('../results/distribution_tokens.png', dpi=300, bbox_inches='tight')
print("\n✓ Graphique sauvegardé : distribution_tokens.png")
plt.show()

# Vocabulaire global
all_tokens = [token for tokens_list in df_clean['tokens'] for token in tokens_list]

print(f"\n=== VOCABULAIRE GLOBAL ===")
print(f"Nombre total de tokens : {len(all_tokens):,}")

# Compter les occurrences
token_counts = Counter(all_tokens)
print(f"Taille du vocabulaire (mots uniques) : {len(token_counts):,}\n")

# Top 30 tokens
print("=== TOP 30 TOKENS LES PLUS FRÉQUENTS ===")
for token, count in token_counts.most_common(30):
    print(f"{token:20s} : {count:7,d}")

# Graphique top 20
top_tokens = token_counts.most_common(20)
tokens_names = [t[0] for t in top_tokens]
tokens_freqs = [t[1] for t in top_tokens]

plt.figure(figsize=(12, 7))
plt.barh(tokens_names[::-1], tokens_freqs[::-1], color='steelblue')
plt.title('Top 20 des tokens les plus fréquents', fontsize=12, fontweight='bold')
plt.xlabel('Fréquence')
plt.ylabel('Token')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('../results/top_tokens.png', dpi=300, bbox_inches='tight')
print("\n Graphique sauvegardé : top_tokens.png")
plt.show()

# Distribution des fréquences
frequencies = list(token_counts.values())

print(f"\n=== DISTRIBUTION DES FRÉQUENCES ===")
print(f"Tokens apparaissant 1 fois       : {sum(1 for c in frequencies if c == 1):,}")
print(f"Tokens apparaissant 2-5 fois     : {sum(1 for c in frequencies if 2 <= c <= 5):,}")
print(f"Tokens apparaissant 6-10 fois    : {sum(1 for c in frequencies if 6 <= c <= 10):,}")
print(f"Tokens apparaissant 11-50 fois   : {sum(1 for c in frequencies if 11 <= c <= 50):,}")
print(f"Tokens apparaissant >50 fois     : {sum(1 for c in frequencies if c > 50):,}")

# Graphique distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(frequencies, bins=100, edgecolor='black', log=True, color='orange')
plt.title('Distribution des fréquences (échelle log)', fontweight='bold')
plt.xlabel('Fréquence')
plt.ylabel('Nombre de tokens (log)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
freq_limited = [f for f in frequencies if f <= 100]
plt.hist(freq_limited, bins=50, edgecolor='black', color='purple', alpha=0.7)
plt.title('Distribution des fréquences (≤ 100)', fontweight='bold')
plt.xlabel('Fréquence')
plt.ylabel('Nombre de tokens')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/distribution_frequences.png', dpi=300, bbox_inches='tight')
print("\n Graphique sauvegardé : distribution_frequences.png")
plt.show()

# Exemples de tokens rares
rare_tokens = [token for token, count in token_counts.items() if count == 1]
print(f"\nExemples de tokens apparaissant 1 seule fois :")
print(rare_tokens[:30])

# Sauvegarde
df_clean.to_csv('../data/movies_preprocessed.csv', index=False)
print(f"\n Dataset preprocessed sauvegardé : ../data/movies_preprocessed.csv")

vocab_df = pd.DataFrame(token_counts.most_common(), columns=['token', 'frequency'])
vocab_df.to_csv('../data/vocabulary.csv', index=False)
print(f" Vocabulaire sauvegardé : ../data/vocabulary.csv ({len(vocab_df)} tokens)\n")

# ===========================
# ÉTAPE 4 : MODÈLE ML CLASSIQUE (SANS TEXTE)
# ===========================

print("\n" + "="*60)
print("=== ÉTAPE 4 : MODÈLE ML CLASSIQUE (FEATURES NON-TEXTUELLES) ===")
print("="*60 + "\n")

# Choisir la variable cible
# On va prédire 'vote_average' (note du film)
target = 'vote_average'

print(f"Variable cible : {target}\n")

# Features numériques disponibles
numeric_features = ['budget', 'popularity', 'runtime', 'vote_count']

# Préparer les données
df_ml = df_clean.copy()

# Supprimer les lignes avec des valeurs manquantes dans les features
print("Avant nettoyage :", len(df_ml))
for col in numeric_features + [target]:
    df_ml = df_ml[df_ml[col].notna()]
print("Après nettoyage :", len(df_ml))

# Supprimer les valeurs aberrantes (budget=0, runtime=0, etc.)
df_ml = df_ml[df_ml['budget'] > 0]
df_ml = df_ml[df_ml['runtime'] > 0]
print("Après suppression des 0 :", len(df_ml))

# Features (X) et target (y)
X = df_ml[numeric_features].values
y = df_ml[target].values

print(f"\nShape de X : {X.shape}")
print(f"Shape de y : {y.shape}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set : {X_train.shape[0]} films")
print(f"Test set  : {X_test.shape[0]} films")

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement du modèle
print("\n=== ENTRAÎNEMENT DU MODÈLE ===")
print("Modèle : Random Forest Regressor")

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

print(" Modèle entraîné !\n")

# Prédictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Évaluation
print("=== ÉVALUATION ===")
print(f"\nTrain set :")
print(f"  R² score : {r2_score(y_train, y_pred_train):.4f}")
print(f"  RMSE     : {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")

print(f"\nTest set :")
print(f"  R² score : {r2_score(y_test, y_pred_test):.4f}")
print(f"  RMSE     : {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")

# Importance des features
feature_importance = pd.DataFrame({
    'feature': numeric_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== IMPORTANCE DES FEATURES ===")
print(feature_importance)

# Graphique importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'], color='coral')
plt.xlabel('Importance')
plt.title('Importance des features (Random Forest)', fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('../results/feature_importance_baseline.png', dpi=300, bbox_inches='tight')
print("\n Graphique sauvegardé : feature_importance_baseline.png")
plt.show()

# Graphique prédictions vs réelles
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Prédiction parfaite')
plt.xlabel('Vote average réel')
plt.ylabel('Vote average prédit')
plt.title('Prédictions vs Valeurs réelles (Test set)', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/predictions_baseline.png', dpi=300, bbox_inches='tight')
print(" Graphique sauvegardé : predictions_baseline.png")
plt.show()

# ===========================
# ÉTAPE 5 : WORD2VEC EMBEDDINGS
# ===========================

print("\n" + "="*60)
print("=== ÉTAPE 5 : APPRENTISSAGE DES EMBEDDINGS WORD2VEC ===")
print("="*60 + "\n")

# Préparer les données pour Word2Vec
# Word2Vec attend une liste de listes de tokens
sentences = df_clean['tokens'].tolist()

print(f"Nombre de descriptions : {len(sentences)}")
print(f"Exemple de phrase (tokens) : {sentences[0][:15]}\n")

# ===========================
# 5.1 : WORD2VEC SANS PREPROCESSING (CORPUS BRUT)
# ===========================

print("=== 5.1 : Word2Vec sur corpus NON traité ===\n")

# Word2Vec basique (pour voir ce qui se passe)
model_raw = Word2Vec(
    sentences=sentences,
    vector_size=100,      # Dimension des vecteurs
    window=5,             # Contexte : 5 mots avant et après
    min_count=1,          # Garder tous les mots (même rares)
    workers=4,            # Parallélisation
    sg=1                  # Skip-gram (meilleur pour petit corpus)
)

print(f" Modèle entraîné !")
print(f"Taille du vocabulaire (min_count=1) : {len(model_raw.wv)}\n")

# Exemples de mots similaires
test_words = ['love', 'action', 'family', 'world']
print("=== Mots similaires (corpus brut) ===")
for word in test_words:
    if word in model_raw.wv:
        similar = model_raw.wv.most_similar(word, topn=5)
        print(f"\n'{word}' est proche de :")
        for sim_word, score in similar:
            print(f"  {sim_word:15s} : {score:.3f}")

print("\n" + "-"*60)

# ===========================
# 5.2 : WORD2VEC AVEC PREPROCESSING (VOCABULAIRE RÉDUIT)
# ===========================

print("\n=== 5.2 : Word2Vec avec vocabulaire réduit ===\n")

# Paramètres pour réduire le vocabulaire
print("Stratégies de réduction du vocabulaire :")
print("1. min_count : supprimer les mots trop rares")
print("2. max_vocab_size : limiter la taille max du vocabulaire")
print("3. Déjà fait : stop words + mots courts supprimés\n")

# Word2Vec avec min_count plus élevé
model_clean = Word2Vec(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=5,          # Supprimer mots apparaissant <5 fois
    workers=4,
    sg=1,
    epochs=10             # Plus d'itérations pour mieux apprendre
)

print(f" Modèle entraîné !")
print(f"Taille du vocabulaire (min_count=5) : {len(model_clean.wv)}")
print(f"Réduction : {len(model_raw.wv)} → {len(model_clean.wv)} mots")
print(f"Pourcentage gardé : {len(model_clean.wv)/len(model_raw.wv)*100:.1f}%\n")

# Exemples de mots similaires (version propre)
print("=== Mots similaires (vocabulaire réduit) ===")
for word in test_words:
    if word in model_clean.wv:
        similar = model_clean.wv.most_similar(word, topn=5)
        print(f"\n'{word}' est proche de :")
        for sim_word, score in similar:
            print(f"  {sim_word:15s} : {score:.3f}")

# Tester des relations
print("\n=== Relations vectorielles ===")
if all(w in model_clean.wv for w in ['man', 'woman', 'king']):
    result = model_clean.wv.most_similar(
        positive=['woman', 'king'],
        negative=['man'],
        topn=3
    )
    print("\nking - man + woman ≈")
    for word, score in result:
        print(f"  {word:15s} : {score:.3f}")

# Visualisation : quelques vecteurs
print("\n=== Exemples de vecteurs ===")
example_words = ['love', 'hero', 'adventure']
for word in example_words:
    if word in model_clean.wv:
        vector = model_clean.wv[word]
        print(f"\n'{word}' → {vector[:10]}... (100 dimensions)")

# Sauvegarde du modèle
model_clean.save("../data/word2vec_model.bin")
print("\n Modèle Word2Vec sauvegardé : ../data/word2vec_model.bin")

print("\n=== REMARQUES ===")
print("""
1. Vocabulaire trop grand (min_count=1) :
   - Beaucoup de mots rares (fautes, noms propres)
   - Embeddings de mauvaise qualité (pas assez d'exemples)
   
2. Vocabulaire réduit (min_count=5) :
   - Meilleure qualité des embeddings
   - Mots fréquents = plus de contexte = meilleurs vecteurs
   
3. Les mots similaires font sens :
   - 'love' proche de 'romance', 'romantic'
   - 'action' proche de 'adventure', 'thriller'
   
4. Prochaine étape : agréger ces vecteurs par film
""")

# ===========================
# ÉTAPE 6 : AGRÉGATION DES EMBEDDINGS
# ===========================

print("\n" + "="*60)
print("=== ÉTAPE 6 : DES EMBEDDINGS AU MODÈLE DE PRÉDICTION ===")
print("="*60 + "\n")

print("=== PROBLÈME ===")
print("""
Chaque film a un nombre différent de mots :
  Film 1 : ["love", "story", "romantic"] → 3 vecteurs de 100 dim
  Film 2 : ["action", "hero", "fight", "save", "world"] → 5 vecteurs de 100 dim
  
Le modèle ML veut un vecteur de TAILLE FIXE pour chaque film !
""")

print("=== SOLUTION : AGRÉGATION ===")
print("""
Options :
1. Moyenne des vecteurs (simple et efficace)
2. Somme des vecteurs
3. Min/Max par dimension
4. TF-IDF pondéré

→ On va utiliser la MOYENNE
""")

# Fonction pour créer le vecteur d'un document (film)
def document_vector(tokens, model):
    """
    Calcule la moyenne des vecteurs des mots d'un document.
    """
    # Filtrer les mots qui sont dans le vocabulaire
    valid_tokens = [token for token in tokens if token in model.wv]
    
    if len(valid_tokens) == 0:
        # Si aucun mot dans le vocabulaire, retourner un vecteur de zéros
        return np.zeros(model.vector_size)
    
    # Moyenne des vecteurs
    vectors = [model.wv[token] for token in valid_tokens]
    return np.mean(vectors, axis=0)

print("\n=== CRÉATION DES VECTEURS DE DOCUMENTS ===\n")

# Créer les vecteurs pour tous les films
doc_vectors = []
for tokens in df_clean['tokens']:
    vec = document_vector(tokens, model_clean)
    doc_vectors.append(vec)

doc_vectors = np.array(doc_vectors)

print(f" Vecteurs créés !")
print(f"Shape : {doc_vectors.shape}")
print(f"  {doc_vectors.shape[0]} films")
print(f"  {doc_vectors.shape[1]} dimensions par film")

# Exemple
print(f"\nExemple - Film 1 : {df_clean['title'].iloc[0]}")
print(f"Tokens : {df_clean['tokens'].iloc[0][:10]}")
print(f"Vecteur : {doc_vectors[0][:10]}... (100 dimensions)")

# Ajouter au dataframe
df_clean['doc_vector'] = list(doc_vectors)

# ===========================
# MODÈLE ML AVEC EMBEDDINGS
# ===========================

print("\n" + "="*60)
print("=== MODÈLE ML AVEC FEATURES TEXTUELLES (EMBEDDINGS) ===")
print("="*60 + "\n")

# Préparer les données
df_ml_text = df_clean.copy()

# Variables
target = 'vote_average'
numeric_features = ['budget', 'popularity', 'runtime', 'vote_count']

# Nettoyage
for col in numeric_features + [target]:
    df_ml_text = df_ml_text[df_ml_text[col].notna()]

df_ml_text = df_ml_text[df_ml_text['budget'] > 0]
df_ml_text = df_ml_text[df_ml_text['runtime'] > 0]

print(f"Nombre de films : {len(df_ml_text)}")

# Features numériques
X_numeric = df_ml_text[numeric_features].values

# Features textuelles (embeddings)
X_text = np.array(df_ml_text['doc_vector'].tolist())

# Combiner les deux
X_combined = np.hstack([X_numeric, X_text])

y = df_ml_text[target].values

print(f"\nShape des features :")
print(f"  Numériques : {X_numeric.shape}")
print(f"  Textuelles : {X_text.shape}")
print(f"  Combinées  : {X_combined.shape}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement
print("\n=== ENTRAÎNEMENT ===")
model_with_text = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_with_text.fit(X_train_scaled, y_train)
print(" Modèle entraîné !\n")

# Prédictions
y_pred_train = model_with_text.predict(X_train_scaled)
y_pred_test = model_with_text.predict(X_test_scaled)

# Évaluation
print("=== ÉVALUATION ===")
print(f"\nTrain set :")
print(f"  R² score : {r2_score(y_train, y_pred_train):.4f}")
print(f"  RMSE     : {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")

print(f"\nTest set :")
r2_with_text = r2_score(y_test, y_pred_test)
rmse_with_text = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"  R² score : {r2_with_text:.4f}")
print(f"  RMSE     : {rmse_with_text:.4f}")

# ===========================
# COMPARAISON BASELINE VS EMBEDDINGS
# ===========================

print("\n" + "="*60)
print("=== COMPARAISON : BASELINE VS AVEC EMBEDDINGS ===")
print("="*60 + "\n")

# Réentraîner baseline sur les mêmes données pour comparaison équitable
X_baseline = X_numeric
X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_baseline, y, test_size=0.2, random_state=42
)

scaler_base = StandardScaler()
X_train_base_scaled = scaler_base.fit_transform(X_train_base)
X_test_base_scaled = scaler_base.transform(X_test_base)

model_baseline = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_baseline.fit(X_train_base_scaled, y_train_base)

y_pred_baseline = model_baseline.predict(X_test_base_scaled)
r2_baseline = r2_score(y_test_base, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test_base, y_pred_baseline))

# Comparaison
comparison = pd.DataFrame({
    'Modèle': ['Baseline (sans texte)', 'Avec embeddings'],
    'R² Score': [r2_baseline, r2_with_text],
    'RMSE': [rmse_baseline, rmse_with_text]
})

print(comparison.to_string(index=False))

print(f"\n=== AMÉLIORATION ===")
r2_improvement = ((r2_with_text - r2_baseline) / abs(r2_baseline)) * 100
rmse_improvement = ((rmse_baseline - rmse_with_text) / rmse_baseline) * 100

print(f"R² : {r2_improvement:+.2f}%")
print(f"RMSE : {rmse_improvement:+.2f}%")

if r2_with_text > r2_baseline:
    print(f"\n Les embeddings AMÉLIORENT le modèle !")
else:
    print(f"\n Les embeddings n'améliorent pas beaucoup...")
    print("Raisons possibles :")
    print("  - Le texte n'est pas très prédictif pour vote_average")
    print("  - Les features numériques sont déjà très bonnes")
    print("  - Corpus trop petit pour Word2Vec")

# Graphique comparaison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Baseline
axes[0].scatter(y_test_base, y_pred_baseline, alpha=0.5, s=20)
axes[0].plot([y_test_base.min(), y_test_base.max()], 
             [y_test_base.min(), y_test_base.max()], 
             'r--', lw=2)
axes[0].set_xlabel('Vote average réel')
axes[0].set_ylabel('Vote average prédit')
axes[0].set_title(f'Baseline (R²={r2_baseline:.4f})', fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Avec embeddings
axes[1].scatter(y_test, y_pred_test, alpha=0.5, s=20, color='green')
axes[1].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2)
axes[1].set_xlabel('Vote average réel')
axes[1].set_ylabel('Vote average prédit')
axes[1].set_title(f'Avec Embeddings (R²={r2_with_text:.4f})', fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/comparison_baseline_embeddings.png', dpi=300, bbox_inches='tight')
print("\n Graphique sauvegardé : comparison_baseline_embeddings.png")
plt.show()