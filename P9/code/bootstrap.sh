#!/bin/bash
# ============================================================
# Bootstrap script EMR — P9 Fruits
# ============================================================
# Ce script s'exécute sur CHAQUE nœud du cluster (master + workers)
# au moment de leur démarrage, AVANT que Spark ne soit lancé.
#
# Objectif : installer toutes les libs Python nécessaires pour
# que la pandas_udf MobileNetV2 puisse tourner sur chaque worker.
# ============================================================

set -e  # arrête le script à la première erreur
set -x  # log toutes les commandes (utile pour debug dans les logs EMR)

# --- 1. Mise à jour de pip ---
sudo python3 -m pip install --upgrade pip

# --- 2. Libs ML / image ---
# Versions épinglées pour reproductibilité.
# tensorflow CPU-only : suffit largement pour de l'inférence MobileNetV2,
# et bien plus léger à installer que tensorflow[and-cuda].
sudo python3 -m pip install \
    "tensorflow-cpu==2.15.0" \
    "Pillow==10.2.0" \
    "numpy<2.0" \
    "pandas==2.2.0" \
    "pyarrow==15.0.0"

# --- 3. Pré-téléchargement des poids MobileNetV2 ---
# Sans ça, chaque worker téléchargerait les poids ImageNet depuis
# storage.googleapis.com au premier appel → lent + risque de timeout
# si plusieurs workers tapent en même temps.
# On le fait UNE FOIS ici, le cache (~/.keras/) sera partagé par tous
# les processus du nœud.
sudo -u hadoop python3 -c "
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
print('MobileNetV2 weights cached')
"

echo "Bootstrap terminé avec succès"
