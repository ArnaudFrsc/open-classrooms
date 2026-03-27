import requests
import pandas as pd
from requests.exceptions import RequestException, Timeout

BASE_URL = "https://world.openfoodfacts.org/cgi/search.pl"
SAVE_PATH = "products_based_on_champagne.csv"
PRODUCT = "champagne"

MAX_RESULTS = 10
PAGE_SIZE = 200 # BATCH SIZE
TIMEOUT_SECONDS = 120

rows = []
page = 1 # BATCHES

while len(rows) < MAX_RESULTS:
    params = {
        "search_terms": PRODUCT,
        "search_simple": 1,
        "action": "process",
        "page_size": PAGE_SIZE,
        "page": page,
        "json": 1
    }

    print("Requesting page :", page)

    # RESPONSE STATUS CODE IF FAIL
    try:
        response = requests.get(BASE_URL, params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
    except Timeout:
        raise RuntimeError("Request timed out while calling OpenFoodFacts API.")
    except RequestException as e:
        raise RuntimeError(f"HTTP request failed: {e}")

    # JSON SAFETY CHECK
    try:
        data = response.json()
    except ValueError:
        raise RuntimeError("API response is not valid JSON.")

    # ADD CHECK ON DATA TO SEE IF (LEN != 0 IN KEY 'products')
    products = data.get("products")
    if not isinstance(products, list):
        raise RuntimeError("Unexpected API format: 'products' key missing or invalid.")

    if len(products) == 0:
        print("No more products returned by the API.")
        break

    for p in products:
        ingredients_text = (p.get("ingredients_text") or "").lower()
        ingredient_tags = p.get("ingredients_tags") or []
        categories_tags = p.get("categories_tags") or []

        # Mentions champagne in ingredients
        mentions_product = (
            PRODUCT in ingredients_text
            or any(PRODUCT in tag for tag in ingredient_tags)
        )

        # Exclude champagne beverages
        is_champagne_beverage = (
            "en:champagnes" in categories_tags
            or "fr:champagnes" in categories_tags
        )

        if mentions_product and not is_champagne_beverage:
            rows.append({
                "foodId": p.get("code"),
                "label": p.get("product_name"),
                "category": p.get("categories"),
                "foodContentsLabel": p.get("ingredients_text"),
                "image": p.get("image_front_url")
            })

        if len(rows) >= MAX_RESULTS:
            break

    page += 1

# CHECK IF DF EMPTY BEFORE SAVING

df = pd.DataFrame(rows)

if df.empty:
    raise RuntimeError("No matching products found. CSV file was not created.")

df.to_csv(SAVE_PATH, index=False, encoding="utf-8")

print(f"File successfully saved: {SAVE_PATH}")
print(f"Total products written: {len(df)}")