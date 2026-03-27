"""
Client Python pour l'API ML — route /predict/stream
Affiche une barre de progression tqdm en temps réel.

Dépendances :
    pip install requests tqdm
"""

import base64
import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

# ─────────────────────────────────────────────
# ⚙️  CONFIGURATION — modifiez ces variables
# ─────────────────────────────────────────────

ROOT_DIR = r'C:\Users\jfurs\Pythonn\OpenClassrooms\DS\P7\data'  # Répertoire de travail pour les fichiers d'entrée/sortie

API_URL     = "http://localhost:8000"   # URL de l'API (remplacez par votre URL Render en prod)
INPUT_FILE  = f'{ROOT_DIR}\\api_test_sample.csv'  # Chemin vers votre fichier CSV ou Excel
OUTPUT_FILE = f'{ROOT_DIR}\\predictions.csv'      # Nom du fichier de sortie (None = nom automatique)

MODEL       = "lgb"                     # Modèle à utiliser : "lgb" ou "xgb"
THRESHOLD   = 0.434                     # Seuil de décision (0.0 → 1.0)
RETURN_PROBA = True                     # True = ajoute une colonne 'proba' dans le résultat

# ─────────────────────────────────────────────
# Appel à l'API avec streaming SSE
# ─────────────────────────────────────────────

def predict_with_progress() -> None:
    filepath = Path(INPUT_FILE)
    if not filepath.exists():
        print(f"❌ Fichier introuvable : {filepath}")
        sys.exit(1)

    url = f"{API_URL.rstrip('/')}/predict/stream"
    params = {
        "model":        MODEL,
        "threshold":    THRESHOLD,
        "return_proba": "true" if RETURN_PROBA else "false",
    }

    print(f"📤 Envoi de '{filepath.name}' vers {url}")
    print(f"   Modèle: {MODEL} | Seuil: {THRESHOLD}\n")

    with open(filepath, "rb") as f:
        response = requests.post(
            url,
            params=params,
            files={"file": (filepath.name, f, _mime(filepath))},
            stream=True,        # indispensable pour lire le SSE au fil de l'eau
            timeout=600,
        )

    if response.status_code != 200:
        print(f"❌ Erreur HTTP {response.status_code} : {response.text}")
        sys.exit(1)

    pbar = None
    out_filename = OUTPUT_FILE

    # Lecture des événements SSE ligne par ligne
    event_name = None
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            event_name = None   # fin d'un bloc SSE
            continue

        if raw_line.startswith("event:"):
            event_name = raw_line.split(":", 1)[1].strip()
            continue

        if raw_line.startswith("data:"):
            payload = json.loads(raw_line.split(":", 1)[1].strip())

            if event_name == "progress":
                total = payload["total"]
                processed = payload["processed"]
                elapsed = payload["elapsed"]

                if pbar is None:
                    pbar = tqdm(
                        total=total,
                        unit="lignes",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                        desc="Prédiction",
                        colour="green",
                    )

                pbar.n = processed
                pbar.set_postfix({"⏱": f"{elapsed}s"})
                pbar.refresh()

            elif event_name == "result":
                if pbar:
                    pbar.n = payload["total"]
                    pbar.refresh()
                    pbar.close()

                # Décodage et sauvegarde du fichier
                file_bytes = base64.b64decode(payload["file_b64"])
                out_filename = out_filename or payload["filename"]
                Path(out_filename).write_bytes(file_bytes)

                print(f"\n✅ Terminé en {payload['elapsed']}s — {payload['total']} lignes traitées")
                print(f"💾 Fichier sauvegardé : {out_filename}")

            elif event_name == "error":
                if pbar:
                    pbar.close()
                print(f"\n❌ Erreur API : {payload['detail']}")
                sys.exit(1)


def _mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "text/csv"
    elif suffix == ".xlsx":
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    return "application/octet-stream"


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    predict_with_progress()
