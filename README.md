# 🎬📱 NLP — Analyse de films et reviews Amazon

Projet universitaire — Master 1 IA & Big Data, Université Paris 8

## 📋 Description

Analyse de textes à l'aide de techniques NLP sur deux datasets distincts :
preprocessing de texte, création de vocabulaire et embeddings **Word2Vec**
pour améliorer la prédiction de variables cibles.

- **Dataset 1** : 45 000 descriptions de films → prédiction de la note moyenne
- **Dataset 2** : 194 000 reviews Amazon d'accessoires téléphoniques → prédiction de la note (1 à 5)

## 🔍 Résultats clés

- Preprocessing complet : tokenisation, suppression des stop words, filtrage
- Vocabulaire analysé avec distribution des fréquences de tokens
- Comparaison modèle baseline (features numériques) vs modèle enrichi (embeddings)
- Les embeddings Word2Vec améliorent la capacité prédictive du modèle

## 🛠️ Stack technique

- **Python** — Pipeline NLP complet
- **Gensim** — Entraînement des embeddings Word2Vec
- **NLTK** — Tokenisation et stop words
- **Scikit-learn** — Modèles de prédiction
- **Pandas / NumPy** — Traitement des données
- **Matplotlib** — Visualisations

## 📁 Structure
```
data/           → Datasets (à télécharger, voir ci-dessous)
src/
├── main.py     → Pipeline NLP — dataset films
└── phones.py   → Pipeline NLP — dataset reviews téléphones
results/        → Graphiques générés
```

## 📥 Récupérer les données

Les fichiers ne sont pas inclus dans ce repo car trop volumineux.

1. `movies_metadata.csv` → [Kaggle - The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
2. `Cell_Phones_and_Accessories_5.json` → [Amazon Reviews Dataset](https://nijianmo.github.io/amazon/index.html)
3. `movies_preprocessed.csv` et `vocabulary.csv` → générés automatiquement par `main.py`
4. `word2vec_model.bin` → généré automatiquement par `main.py`
5. `word2vec_phones.bin` → généré automatiquement par `phones.py`

Place les fichiers téléchargés dans le dossier `data/`.

## 🚀 Lancer le projet
```bash
# Installer les dépendances
pip install pandas numpy matplotlib scikit-learn gensim nltk

# Télécharger les ressources NLTK
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Lancer le pipeline films
python src/main.py

# Lancer le pipeline reviews téléphones
python src/phones.py
```

## 📊 Visualisations générées

### Dataset films (`main.py`)
- `distributions_variables_cibles.png` — Distribution des variables cibles
- `distribution_tokens.png` — Distribution des tokens par description
- `distribution_frequences.png` — Fréquence des tokens
- `top_tokens.png` — Top tokens les plus fréquents
- `feature_importance_baseline.png` — Importance des features (baseline)
- `predictions_baseline.png` — Prédictions vs valeurs réelles
- `comparison_baseline_embeddings.png` — Comparaison baseline vs Word2Vec

### Dataset reviews téléphones (`phones.py`)
- `overall_distribution.png` — Distribution des notes (1 à 5)