"""Extraction des 10 premiers produits 'champagne' d'Open Food Facts vers un CSV."""
 
import csv
import sys
import requests
 
URL = "https://search.openfoodfacts.org/search"
PARAMS = {"q": "champagne", "page_size": 10, "langs": "fr"}
HEADERS = {"User-Agent": "EpicerieFine/1.0 (contact@exemple.com)"}

try:
    response = requests.get(URL, params=PARAMS, headers=HEADERS, timeout=30)
    response.raise_for_status()
    products = response.json()["hits"]
except requests.RequestException as e:
    sys.exit(f"Erreur lors de l'appel à l'API : {e}")
except (ValueError, KeyError) as e:
    sys.exit(f"Réponse API invalide : {e}")

with open("champagne_products.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["foodId", "label", "category", "foodContentsLabel", "image"])
    for p in products:
        print("Exporting product:", p.get("code",  "N/A"))
        writer.writerow([
            p.get("code", ""),
            p.get("product_name", ""),
            p.get("categories", ""),
            p.get("ingredients_text", ""),
            p.get("image_url", ""),
        ])

print(f"{len(products)} produits exportés dans champagne_products.csv")