# ML Prediction API

API FastAPI pour prédire un label **0 ou 1** à partir d'un fichier CSV ou Excel.  
Deux modèles disponibles : **LightGBM** et **XGBoost**.

---

## Structure du projet

```
project/
│
├── main.py               ← API FastAPI
├── requirements.txt      ← Dépendances Python
├── README.md             ← Ce fichier
│
└── models/               ← ⚠️ À créer — placez vos .pkl ici
    ├── LightGBM_best_model.pkl
    └── XGBoost_best_model.pkl
```

> **Important** : le dossier `models/` n'est pas inclus dans ce repo.  
> Copiez vos fichiers `.pkl` générés par MLflow dans ce dossier avant de démarrer.

---

## Lancement en local

### 1. Prérequis

- Python 3.10+
- pip

### 2. Installation

```bash
# Créer et activer un environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Placer les modèles

```bash
mkdir models
cp /chemin/vers/LightGBM_best_model.pkl models/
cp /chemin/vers/XGBoost_best_model.pkl  models/
```

### 4. Démarrer l'API

```bash
uvicorn main:app --reload --port 8000
```

L'API est disponible sur : **http://localhost:8000**  
Documentation interactive Swagger : **http://localhost:8000/docs**

---

## Utilisation de l'API

### `GET /` — Statut

Vérifie que l'API tourne et liste les modèles disponibles.

```bash
curl http://localhost:8000/
```

```json
{
  "status": "ok",
  "available_models": ["lgb", "xgb"],
  "message": "POST /predict pour effectuer des prédictions."
}
```

---

### `POST /predict` — Prédiction

**Paramètres (query string) :**

| Paramètre      | Type    | Défaut | Description                                                  |
|----------------|---------|--------|--------------------------------------------------------------|
| `model`        | string  | `lgb`  | Modèle à utiliser : `lgb` (LightGBM) ou `xgb` (XGBoost)    |
| `threshold`    | float   | `0.5`  | Seuil de décision pour label=1. Baissez pour + de recall.   |
| `return_proba` | boolean | `true` | Si `true`, ajoute une colonne `proba` (proba classe 1)       |

**Corps de la requête :** fichier `.csv`, `.xlsx` ou `.xls` (multipart/form-data)

**Réponse :** le fichier original enrichi de deux colonnes :
- `predicted_label` : `0` ou `1`
- `proba` : probabilité d'appartenir à la classe 1 (si `return_proba=true`)

---

#### Exemples avec curl

```bash
# CSV avec LightGBM (défaut)
curl -X POST "http://localhost:8000/predict" \
  -F "file=@mon_fichier.csv" \
  --output predictions.csv

# Excel avec XGBoost, seuil à 0.3
curl -X POST "http://localhost:8000/predict?model=xgb&threshold=0.3" \
  -F "file=@mon_fichier.xlsx" \
  --output predictions.xlsx

# Sans colonne proba
curl -X POST "http://localhost:8000/predict?return_proba=false" \
  -F "file=@mon_fichier.csv" \
  --output predictions.csv
```

## Déploiement sur Render

### 1. Préparer le repo Git

```bash
git init
git add main.py requirements.txt README.md
git commit -m "Initial API"
git remote add origin https://github.com/votre-username/votre-repo.git
git push -u origin main
```

> ⚠️ **Ne commitez pas vos fichiers `.pkl`** si leur taille dépasse quelques dizaines de Mo.  
> Utilisez Git LFS ou la solution alternative décrite ci-dessous.

### 2. Gérer les modèles `.pkl` sur Render

**Option A — Modèles petits (< 100 Mo) : commit direct**
```bash
git add models/
git commit -m "Add models"
git push
```

### 3. Créer le service sur Render

1. Allez sur [render.com](https://render.com) → **New → Web Service**
2. Connectez votre repo GitHub
3. Renseignez les paramètres :

| Paramètre        | Valeur                                          |
|------------------|-------------------------------------------------|
| **Runtime**      | Python 3                                        |
| **Build Command**| `pip install -r requirements.txt`               |
| **Start Command**| `uvicorn main:app --host 0.0.0.0 --port $PORT` |

4. **Deploy** → votre API sera accessible sur `https://votre-service.onrender.com`

### 4. Variable d'environnement (optionnel)

Si vos modèles sont dans un sous-dossier différent, définissez :

| Clé          | Valeur      |
|--------------|-------------|
| `MODELS_DIR` | `models`    |

---

## Format du fichier d'entrée

- Les noms de colonnes doivent correspondre aux features utilisées à l'entraînement.  
- Les caractères spéciaux dans les noms de colonnes sont automatiquement remplacés par `_` (même logique qu'à l'entraînement).
- Les valeurs manquantes sont acceptées — le modèle les gère nativement (LightGBM/XGBoost).
- La colonne `TARGET` ne doit **pas** être présente dans le fichier (c'est ce qu'on prédit).

### Exemple de fichier d'entrée (`input.csv`)

```
feature_1,feature_2,feature_3,...
0.5,1.2,NaN,...
1.1,0.3,2.4,...
```

### Exemple de fichier de sortie (`input_predictions.csv`)

```
feature_1,feature_2,feature_3,...,predicted_label,proba
0.5,1.2,NaN,...,0,0.1823
1.1,0.3,2.4,...,1,0.7341
```

---

## Codes d'erreur

| Code | Signification                                              |
|------|------------------------------------------------------------|
| 400  | Modèle inconnu                                             |
| 415  | Format de fichier non supporté (uniquement csv/xlsx/xls)  |
| 422  | Fichier vide ou colonnes manquantes                        |
| 500  | Erreur interne lors de la prédiction                       |

---

## Notes techniques

- Le seuil par défaut est `0.5`. Dans votre cas métier, vous avez optimisé un seuil différent lors de l'entraînement — renseignez-le via le paramètre `threshold` pour reproduire exactement les résultats d'entraînement.
- L'API charge les modèles **une seule fois au démarrage** pour des performances optimales.
- Les fichiers uploadés ne sont jamais écrits sur le disque — tout est traité en mémoire.
