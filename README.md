<<<<<<< HEAD
# Projet ML Retail - Analyse Comportementale Clientèle

## Description
Projet de machine learning pour analyser le comportement des clients d'un e-commerce de cadeaux.  
Objectif : prédire le churn, segmenter les clients et déployer un modèle via Flask.

## Installation

```bash
# 1. Créer l'environnement virtuel
python -m venv venv

# 2. Activer (Windows)
venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

## Structure du projet

```
projet_ml_retail/
├── data/
│   ├── raw/            # Données brutes originales
│   ├── processed/      # Données nettoyées
│   └── train_test/     # Données splittées (train/test)
├── notebooks/          # Notebooks Jupyter (prototypage)
├── src/                # Scripts Python (production)
│   ├── utils.py        # Fonctions utilitaires
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── models/             # Modèles sauvegardés (.pkl, .joblib)
├── app/                # Application web Flask
├── reports/            # Rapports et visualisations
├── requirements.txt
├── .gitignore
└── README.md
```

## Guide d'utilisation

```bash
# 1. Placer les données brutes dans data/raw/clients.csv

# 2. Prétraitement
python src/preprocessing.py

# 3. Entraînement des modèles
python src/train_model.py

# 4. Prédiction (test)
python src/predict.py

# 5. Lancer l'application Flask
python app/app.py
```
=======
# Machine_Learning
>>>>>>> bc8ebb5ccafe69e2a119f215a8ebf274fa6bff96
