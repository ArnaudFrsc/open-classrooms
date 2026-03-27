import sqlite3
import pandas as pd

FP = r"C:\Users\jfurs\@Python\OpenClassrooms\DS\P5\olist.db"

########## Requests to a SQLite Database asked by Fernanda ##########

"""
En excluant les commandes annulées, quelles sont les commandes récentes 
de moins de 3 mois que les clients ont reçues avec au moins 3 jours de retard ?
"""

"""
SELECT *
FROM (
    SELECT 
        o.order_id,
        o.customer_id,
        o.order_purchase_timestamp,
        o.order_estimated_delivery_date,
        o.order_delivered_customer_date,
        julianday(o.order_delivered_customer_date) - julianday(o.order_estimated_delivery_date) AS delay_days
    FROM orders o
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp >= date('2018-10-17', '-3 months') -- CHANGE DATE
) sub
WHERE sub.delay_days >= 3
ORDER BY sub.order_purchase_timestamp DESC;
"""

#####################################################################

"""
Qui sont les vendeurs ayant généré un chiffre d'affaires de plus de 100.000 Real sur des commandes livrées via Olist ?
"""

"""
SELECT 
    s.seller_id,
    s.seller_city,
    s.seller_state,
    SUM(oi.price + oi.freight_value) AS total_revenue
FROM order_items oi
JOIN orders o ON oi.order_id = o.order_id
JOIN sellers s ON oi.seller_id = s.seller_id
WHERE o.order_status IN ('delivered', "approved", "shipped")
GROUP BY s.seller_id, s.seller_city, s.seller_state
HAVING total_revenue > 100000
ORDER BY total_revenue DESC;
"""

#####################################################################

"""Qui sont les nouveaux vendeurs (moins de 3 mois d'ancienneté) 
 qui sont déjà très engagés avec la plateforme (ayant déjà vendu plus de 30 produits) ?"""

"""
WITH seller_stats AS (
    SELECT 
        oi.seller_id,
        MIN(o.order_purchase_timestamp) AS first_sale_date,
        COUNT(oi.order_item_id) AS total_products_sold
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered' -- ADD IN STAEMENT
    GROUP BY oi.seller_id
)
SELECT
    s.seller_id,
    s.seller_city,
    s.seller_state,
    ss.first_sale_date,
    ss.total_products_sold
FROM seller_stats ss
JOIN sellers s ON ss.seller_id = s.seller_id
WHERE ss.first_sale_date >= date('2018-10-17', '-3 months') -- CHANGE DATE CTE
  AND ss.total_products_sold > 30
ORDER BY ss.total_products_sold DESC;
"""

#####################################################################

"""Quels sont les 5 codes postaux, enregistrant plus de 30 reviews, avec le pire review score moyen sur les 12 derniers mois ?"""

"""
SELECT 
    c.customer_zip_code_prefix AS zip_code,
    AVG(r.review_score) AS avg_score,
    COUNT(r.review_id) AS total_reviews,
    MIN(r.review_creation_date) AS oldest_review_in_last_12_months,
    MAX(r.review_creation_date) AS latest_review_in_last_12_months
FROM order_reviews r
JOIN orders o ON r.order_id = o.order_id
JOIN customers c ON o.customer_id = c.customer_id
WHERE r.review_creation_date BETWEEN date('2018-10-17','-12 months') AND '2018-10-17' -- CHANGE DATE CTE
GROUP BY c.customer_zip_code_prefix
HAVING COUNT(r.review_id) > 30
ORDER BY avg_score ASC
LIMIT 5;
"""

#######################################################################

########## DB Requests to get clustering data ##########

# Path SQLite database file
db_path = FP

# SQL query
query = """
WITH order_money AS (
    SELECT 
        oi.order_id,
        SUM(oi.price + oi.freight_value) AS order_total
    FROM order_items oi
    GROUP BY oi.order_id
),
order_reviews_summary AS (
    SELECT 
        r.order_id,
        r.review_score,
        r.review_id
    FROM order_reviews r
)
SELECT
    o.order_id,
    o.order_status,
    o.order_purchase_timestamp AS order_date,
    o.order_delivered_customer_date AS delivery_date,
    o.order_estimated_delivery_date,
    om.order_total,
    ors.review_score,
    c.customer_id,
    c.customer_unique_id,
    c.customer_city,
    c.customer_state
FROM orders o
LEFT JOIN order_money om ON o.order_id = om.order_id
LEFT JOIN order_reviews_summary ors ON o.order_id = ors.order_id
LEFT JOIN customers c ON o.customer_id = c.customer_id
WHERE o.order_status IN ('delivered', 'invoiced', 'created', 'shipped', 'approved');
"""

conn = sqlite3.connect(db_path)

# Load results into DF
df = pd.read_sql_query(query, conn)

conn.close()

# Save DataFrame to CSV
output_file = "query_results.csv"
df.to_csv(output_file, index=False)

print(f"Data saved to {output_file}")