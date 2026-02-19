Projet GitHub : https://github.com/Arthur94R/m1-nlp-word2vec

# 🎬📱 TP1 — Embeddings Word2Vec

Projet universitaire — Master 1 IA & Big Data, Université Paris 8

## 📋 Description

Création et analyse d'embeddings Word2Vec sur deux datasets textuels :
- **Dataset 1** : 45 466 descriptions de films
- **Dataset 2** : 194 439 reviews Amazon d'accessoires téléphoniques

L'objectif est de comprendre comment Word2Vec capture le sens sémantique des mots en transformant du texte en vecteurs numériques.

## 🎯 Objectif

Démontrer que Word2Vec crée des représentations vectorielles qui capturent :
- La **similarité sémantique** (mots similaires → vecteurs proches)
- Les **relations complexes** (analogies comme king - man + woman ≈ queen)
- Le **contexte d'utilisation** des mots

## 🔍 Étapes du TP

1. **Chargement et analyse** des datasets
2. **Preprocessing** : lowercase, tokenisation, suppression stop words
3. **Réduction du vocabulaire** : min_count=5 pour garder les mots fréquents
4. **Entraînement Word2Vec** : Skip-gram, 100 dimensions
5. **Analyse des embeddings** :
   - Mots similaires
   - Relations vectorielles
   - Visualisation des vecteurs

## 🛠️ Stack technique

- **Python 3.13** — Langage principal
- **Gensim** — Entraînement Word2Vec
- **NLTK** — Tokenisation et stop words
- **Pandas / NumPy** — Traitement des données

## 📁 Structure
```
data/              → Datasets (à télécharger)
src/
├── main.py        → Pipeline Word2Vec — films
└── phones.py      → Pipeline Word2Vec — reviews téléphones
results/           → Modèles sauvegardés
```

## 📥 Récupérer les données

**Datasets à télécharger :**

1. **movies_metadata.csv** → [Kaggle - The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
2. **Cell_Phones_and_Accessories_5.json** → [Amazon Reviews](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz)

Placer les fichiers dans le dossier `data/`.

**Fichiers générés automatiquement :**
- `word2vec_films.bin` — Modèle Word2Vec entraîné sur les films
- `word2vec_phones.bin` — Modèle Word2Vec entraîné sur les reviews

## 🚀 Installation et lancement

### Installation
```bash
# Installer les dépendances
pip install pandas numpy gensim nltk

# Télécharger les ressources NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Lancement
```bash
# Pipeline films
python src/main.py

# Pipeline reviews téléphones
python src/phones.py
```

## 📊 Résultats attendus

### Mots similaires (dataset films)
```
'love' est proche de :
  affection    : 0.741
  romance      : 0.735
  madly        : 0.730

'action' est proche de :
  installment  : 0.808
  paced        : 0.786
  thriller     : 0.774
```

### Relations vectorielles
```
king - man + woman ≈ princess, ruler, empress
```

### Vecteurs
Chaque mot = vecteur de 100 dimensions
```
'love' → [0.084, 0.115, -0.090, -0.551, ...]
'hero' → [0.462, -0.068, 0.424, -0.315, ...]
```

## 📝 Livrables

- ✅ Code source (`main.py`, `phones.py`)
- ✅ Modèles Word2Vec entraînés
- ✅ Rapport PDF d'analyse
- ✅ README

## 🎓 Concepts clés

- **Word2Vec** : Algorithme de représentation textuelle (pas un modèle de prédiction)
- **Skip-gram** : Méthode qui prédit le contexte à partir d'un mot
- **Embeddings** : Représentations vectorielles denses des mots
- **Similarité cosinus** : Mesure de proximité entre vecteurs