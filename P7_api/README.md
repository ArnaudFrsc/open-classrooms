# ML Prediction API

API FastAPI pour prédire un label **0 ou 1** à partir d'un fichier CSV ou Excel.  
Modèle : **LightGBM**.

La colonne **`SK_ID_CURR`** est toujours conservée dans le fichier de sortie et placée en première position.

---

## Structure du projet

```
project/
│
├── main.py               ← API FastAPI
├── requirements.txt      ← Dépendances Python
├── tests_unitaires.py    ← Tests pytest
├── README.md             ← Ce fichier
│
└── models/               ← ⚠️ À créer — placez votre .pkl ici
    └── LightGBM_best_model.pkl
```

> **Important** : le dossier `models/` n'est pas inclus dans ce repo.  
> Copiez votre fichier `LightGBM_best_model.pkl` généré par MLflow dans ce dossier avant de démarrer.

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

### 3. Placer le modèle

```bash
mkdir models
cp /chemin/vers/LightGBM_best_model.pkl models/
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

Vérifie que l'API tourne et liste le modèle disponible.

```bash
curl http://localhost:8000/
```

```json
{
  "status": "ok",
  "available_models": ["lgb"],
  "id_column_preserved": "SK_ID_CURR",
  "routes": {
    "POST /predict/explain": "Prédiction + top-N SHAP values par client"
  }
}
```

---

### `GET /models` — Modèle disponible

Liste le modèle chargé et la disponibilité de l'explainer SHAP.

```bash
curl http://localhost:8000/models
```

```json
{
  "available_models": ["lgb"],
  "shap_available": ["lgb"]
}
```

---

### `POST /predict/explain` — Prédiction + analyse SHAP locale

**Paramètres (query string) :**

| Paramètre      | Type    | Défaut | Description                                              |
|----------------|---------|--------|----------------------------------------------------------|
| `threshold`    | float   | `0.5`  | Seuil de décision pour label=1                           |
| `return_proba` | boolean | `true` | Ajoute la colonne `proba` (probabilité classe 1)         |
| `n_top`        | int     | `10`   | Nombre de features SHAP à inclure dans le fichier (1–50) |

**Corps de la requête :** fichier `.csv`, `.xlsx` ou `.xls` (multipart/form-data)

**Réponse :** fichier enrichi avec les colonnes :
- `SK_ID_CURR` : identifiant client (en première position)
- `predicted_label` : `0` ou `1`
- `proba` : probabilité classe 1 (si `return_proba=true`)
- `shap_<feature>` × n_top : valeur SHAP des features les plus importantes

**Interprétation des valeurs SHAP :**
- Valeur **positive** → la feature pousse la prédiction vers la classe **1** (défaut de paiement)
- Valeur **négative** → la feature pousse la prédiction vers la classe **0** (non-défaut)
- Plus la valeur absolue est grande, plus la feature a influencé la décision

> ⚠️ Le calcul SHAP (`TreeExplainer`) est intensif.  
> Pour de gros fichiers (> 10 000 lignes), préférez traiter par batch côté client.

#### Exemple de fichier de sortie (`_explained.csv`)

```
SK_ID_CURR,feature_1,...,predicted_label,proba,shap_EXT_SOURCE_3,shap_EXT_SOURCE_2,...
100001,0.5,...,0,0.1823,-0.4521,-0.2134,...
100002,1.1,...,1,0.7341,0.3812,0.2901,...
```

#### Exemple curl

```bash
curl -X POST "http://localhost:8000/predict/explain?n_top=10" \
  -F "file=@mon_fichier.csv" \
  --output explained.csv
```

---

## Utilisation dans Streamlit

Le fichier de sortie enrichi (`_explained.csv`) peut être directement utilisé dans une interface Streamlit grâce à `SK_ID_CURR` :

```python
import pandas as pd

df = pd.read_csv("predictions_explained.csv")

# Recherche par ID client
client_id = 100001
client = df[df["SK_ID_CURR"] == client_id].iloc[0]

# Colonnes SHAP disponibles
shap_cols = [c for c in df.columns if c.startswith("shap_")]
shap_values = client[shap_cols]
```

---

## Déploiement sur Render

### 1. Préparer le repo Git

```bash
git init
git add main.py requirements.txt tests_unitaires.py README.md
git commit -m "Initial API v2.0"
git remote add origin https://github.com/votre-username/votre-repo.git
git push -u origin main
```

### 2. Gérer le modèle `.pkl` sur Render

**Option A — Modèle petit (< 100 Mo) : commit direct**
```bash
git add models/
git commit -m "Add model"
git push
```

### 3. Créer le service sur Render

1. Allez sur [render.com](https://render.com) → **New → Web Service**
2. Connectez votre repo GitHub
3. Renseignez les paramètres :

| Paramètre         | Valeur                                          |
|-------------------|-------------------------------------------------|
| **Runtime**       | Python 3                                        |
| **Build Command** | `pip install -r requirements.txt`               |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

4. **Deploy** → votre API sera accessible sur `https://votre-service.onrender.com`

### 4. Variables d'environnement (optionnel)

| Clé          | Valeur   | Description                |
|--------------|----------|----------------------------|
| `MODELS_DIR` | `models` | Dossier contenant le .pkl  |

---

## Format du fichier d'entrée

- La colonne `SK_ID_CURR` doit être présente — elle sera toujours conservée.
- Les noms de colonnes sont normalisés automatiquement (caractères spéciaux → `_`).
- Les valeurs manquantes sont acceptées — LightGBM les gère nativement.
- La colonne `TARGET` ne doit **pas** être présente (c'est ce qu'on prédit).

### Exemple de fichier d'entrée

```
SK_ID_CURR,feature_1,feature_2,feature_3,...
100001,0.5,1.2,NaN,...
100002,1.1,0.3,2.4,...
```

### Exemple de fichier de sortie

```
SK_ID_CURR,feature_1,...,predicted_label,proba,shap_EXT_SOURCE_3,shap_EXT_SOURCE_2,...
100001,0.5,...,0,0.1823,-0.4521,-0.2134,...
100002,1.1,...,1,0.7341,0.3812,0.2901,...
```

---

## Codes d'erreur

| Code | Signification                                              |
|------|------------------------------------------------------------|
| 415  | Format de fichier non supporté (uniquement csv/xlsx/xls)  |
| 422  | Fichier vide ou colonnes manquantes                        |
| 500  | Erreur interne lors de la prédiction ou du calcul SHAP     |
| 503  | Explainer SHAP non disponible                              |

---

## Notes techniques

- Le seuil par défaut est `0.5`. Renseignez votre seuil optimisé via le paramètre `threshold`.
- Le modèle et l'explainer SHAP sont chargés **une seule fois au démarrage**.
- Les fichiers uploadés ne sont jamais écrits sur le disque — tout est traité en mémoire.
- Les top-N features SHAP sont sélectionnées par **|mean SHAP| décroissant** sur l'ensemble du batch envoyé — ce sont les features les plus globalement influentes pour ce groupe de clients.
