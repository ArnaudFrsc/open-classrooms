"""
Dashboard de scoring crédit — Home Credit
Script Streamlit unifié : upload du fichier brut, appel API /predict/explain,
cache des résultats par client dans session_state.
"""

import io
import re

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from config import (
    API_COLS, API_URL, MODEL, DEFAULT_THRESHOLD, DEFAULT_N_TOP_SHAP,
    FEATURE_LABELS,
    COLOR_ACCEPT, COLOR_REJECT, COLOR_HIGHLIGHT,
    PLOTLY_LAYOUT, DEFAULT_MARGIN, DEFAULT_XAXIS, DEFAULT_YAXIS,
    GLOBAL_CSS,
    html_sidebar_cache_summary, html_sidebar_status, html_verdict_card,
    HTML_LANDING_INSTRUCTIONS, HTML_TRAIN_MISSING, HTML_FOOTER,
)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────


def get_label(col: str) -> str:
    return FEATURE_LABELS.get(col, col.replace("_", " ").title())


def feature_name(shap_col: str) -> str:
    return re.sub(r"^shap_", "", shap_col)


def feature_label(shap_col: str) -> str:
    return get_label(feature_name(shap_col))


def align_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Aligne le DataFrame sur les colonnes attendues par l'API."""
    df.columns = df.columns.str.replace(r'[^0-9a-zA-Z_]', '_', regex=True)
    return df.reindex(columns=API_COLS)


def call_api_explain(row_df: pd.DataFrame, threshold: float, n_top: int) -> dict | None:
    """
    Envoie un CSV d'une seule ligne à /predict/explain et retourne le résultat parsé.
    Retourne None en cas d'erreur.
    """
    csv_buffer = io.StringIO()
    row_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    url = f"{API_URL.rstrip('/')}/predict/explain"
    params = {
        "model": MODEL,
        "threshold": threshold,
        "return_proba": "true",
        "n_top": n_top,
    }

    try:
        resp = requests.post(
            url,
            params=params,
            files={"file": ("client.csv", csv_bytes, "text/csv")},
            timeout=300,
        )
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Erreur réseau : {e}")
        return None

    if resp.status_code != 200:
        st.error(f"❌ Erreur API ({resp.status_code}) : {resp.text[:500]}")
        return None

    try:
        result_df = pd.read_csv(io.BytesIO(resp.content))
    except Exception as e:
        st.error(f"❌ Impossible de lire la réponse API : {e}")
        return None

    if result_df.empty:
        st.error("❌ L'API a retourné un résultat vide.")
        return None

    row = result_df.iloc[0]
    shap_cols = [c for c in result_df.columns if c.startswith("shap_")]
    shap_values = {c: float(row[c]) for c in shap_cols}

    return {
        "predicted_label": int(row["predicted_label"]),
        "proba": float(row["proba"]),
        "shap_values": shap_values,
    }


# ─────────────────────────────────────────────
# Config page + CSS
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Scoring Crédit · Home Credit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────

if "results_cache" not in st.session_state:
    st.session_state.results_cache = {}

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

if "file_id" not in st.session_state:
    st.session_state.file_id = None

if "df_train" not in st.session_state:
    st.session_state.df_train = None

if "train_file_id" not in st.session_state:
    st.session_state.train_file_id = None

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="main-title">🏦 Scoring<br>Crédit</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Home Credit · Analyse client</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Charger le fichier de données clients",
        type=["csv", "xlsx"],
        help="Fichier CSV ou Excel contenant au minimum la colonne SK_ID_CURR",
    )

    if uploaded:
        new_file_id = f"{uploaded.name}_{uploaded.size}"
        if new_file_id != st.session_state.file_id:
            st.session_state.file_id = new_file_id
            st.session_state.results_cache = {}
            if uploaded.name.endswith(".csv"):
                st.session_state.df_raw = pd.read_csv(uploaded)
            else:
                st.session_state.df_raw = pd.read_excel(uploaded)

    uploaded_train = st.file_uploader(
        "Charger le jeu d'entraînement (référence)",
        type=["csv", "xlsx"],
        help="Fichier _explained.csv du training set (précalculé via predict_client.py) — colonnes : SK_ID_CURR, proba, predicted_label, shap_*",
        key="train_uploader",
    )

    if uploaded_train:
        new_train_id = f"{uploaded_train.name}_{uploaded_train.size}"
        if new_train_id != st.session_state.train_file_id:
            st.session_state.train_file_id = new_train_id
            if uploaded_train.name.endswith(".csv"):
                st.session_state.df_train = pd.read_csv(uploaded_train)
            else:
                st.session_state.df_train = pd.read_excel(uploaded_train)

    df_raw = st.session_state.df_raw

    if df_raw is not None:
        if "SK_ID_CURR" not in df_raw.columns:
            st.error("❌ Colonne SK_ID_CURR absente du fichier.")
            st.stop()

        st.markdown("---")
        st.markdown('<p class="metric-label">Fichier chargé</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{len(df_raw):,}</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">clients</p>', unsafe_allow_html=True)

        # Résumé du cache
        n_cached = len(st.session_state.results_cache)
        if n_cached > 0:
            n_acc = sum(1 for v in st.session_state.results_cache.values() if v["predicted_label"] == 0)
            n_rej = n_cached - n_acc
            st.markdown(html_sidebar_cache_summary(n_cached, n_acc, n_rej), unsafe_allow_html=True)

        st.markdown("---")

        # Sélection client
        client_ids = sorted(df_raw["SK_ID_CURR"].unique().tolist())
        selected_id = st.selectbox(
            "🔍 Sélectionner un client",
            options=client_ids,
            format_func=lambda x: f"#{int(x)}",
        )

        # Paramètres API
        st.markdown("---")
        st.markdown('<p class="metric-label">Paramètres</p>', unsafe_allow_html=True)
        threshold = st.slider("Seuil de décision", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
        n_top_shap = st.slider("Nombre de features SHAP", 3, 20, DEFAULT_N_TOP_SHAP)

        # Bouton d'analyse
        already_cached = selected_id in st.session_state.results_cache
        btn_label = "✅ Déjà analysé — Réafficher" if already_cached else "🚀 Lancer l'analyse"
        analyze = st.button(btn_label, use_container_width=True, type="primary")

        st.markdown(html_sidebar_status(threshold), unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main — état initial
# ─────────────────────────────────────────────

if df_raw is None:
    st.markdown('<h1 class="main-title">Analyse de<br><em>scoring crédit</em></h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Chargez un fichier de données dans la barre latérale</p>', unsafe_allow_html=True)
    st.markdown(HTML_LANDING_INSTRUCTIONS, unsafe_allow_html=True)
    st.stop()

if not analyze:
    if selected_id not in st.session_state.results_cache:
        st.markdown('<h1 class="main-title">Analyse de<br><em>scoring crédit</em></h1>', unsafe_allow_html=True)
        st.markdown('<p class="main-subtitle">Sélectionnez un client et cliquez sur « Lancer l\'analyse »</p>', unsafe_allow_html=True)
        st.stop()

# ─────────────────────────────────────────────
# Appel API (ou lecture cache)
# ─────────────────────────────────────────────

if selected_id not in st.session_state.results_cache:
    with st.spinner("⏳ Appel API en cours — calcul SHAP…"):
        client_row = df_raw[df_raw["SK_ID_CURR"] == selected_id].copy()
        client_row_aligned = align_columns(client_row)

        result = call_api_explain(client_row_aligned, threshold, n_top_shap)
        if result is None:
            st.stop()

        st.session_state.results_cache[selected_id] = result
        st.rerun()

# Lire depuis le cache
cached = st.session_state.results_cache[selected_id]
label = cached["predicted_label"]
proba = cached["proba"]
shap_values = cached["shap_values"]
is_accepted = label == 0

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown(f'<h1 class="main-title">Client <em>#{int(selected_id)}</em></h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Analyse individuelle · Scoring crédit</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Row 1 — Verdict + Jauge
# ─────────────────────────────────────────────

col_verdict, col_metrics = st.columns([1.4, 1], gap="large")

with col_verdict:
    st.markdown(html_verdict_card(is_accepted), unsafe_allow_html=True)

with col_metrics:
    proba_pct = round(proba * 100, 1)
    risk_color = COLOR_REJECT if proba >= threshold else COLOR_ACCEPT

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba_pct,
        number={"suffix": "%", "font": {"size": 28, "family": "DM Serif Display", "color": "#f0ede6"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4b5563", "tickfont": {"size": 9}},
            "bar": {"color": risk_color, "thickness": 0.25},
            "bgcolor": "#161920",
            "borderwidth": 0,
            "steps": [
                {"range": [0, threshold * 100], "color": "#0a1f14"},
                {"range": [threshold * 100, 100], "color": "#1f0a0a"},
            ],
            "threshold": {
                "line": {"color": COLOR_HIGHLIGHT, "width": 2},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
        title={"text": "PROBABILITÉ DE DÉFAUT", "font": {"size": 9, "color": "#6b7280", "family": "DM Mono"}},
    ))
    fig_gauge.update_layout(**PLOTLY_LAYOUT, height=200)
    fig_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ─────────────────────────────────────────────
# Row 2 — SHAP waterfall
# ─────────────────────────────────────────────

if shap_values:
    st.markdown('<p class="section-title">📊 Analyse locale SHAP — contribution des features</p>', unsafe_allow_html=True)

    shap_display = {feature_label(c): v for c, v in shap_values.items()}
    shap_sorted = dict(sorted(shap_display.items(), key=lambda x: x[1]))

    features = list(shap_sorted.keys())
    values = list(shap_sorted.values())
    colors = [COLOR_REJECT if v > 0 else COLOR_ACCEPT for v in values]

    fig_shap = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>SHAP : %{x:.4f}<extra></extra>",
    ))
    fig_shap.add_vline(x=0, line_color="#4b5563", line_width=1)
    fig_shap.update_layout(
        **PLOTLY_LAYOUT,
        margin=DEFAULT_MARGIN,
        xaxis=DEFAULT_XAXIS,
        yaxis=dict(gridcolor="#1e2028", zerolinecolor="#2d3040", tickfont=dict(size=10)),
        height=420,
        title=dict(
            text="<span style='font-size:10px;color:#6b7280'>Rouge = pousse vers DÉFAUT · Vert = pousse vers NON-DÉFAUT</span>",
            x=0, font=dict(size=10),
        ),
        xaxis_title="Valeur SHAP",
    )
    st.plotly_chart(fig_shap, use_container_width=True)
else:
    st.info("Aucune valeur SHAP retournée par l'API.")

# ─────────────────────────────────────────────
# Row 3 — Distribution des probabilités (training set)
# ─────────────────────────────────────────────

df_train = st.session_state.df_train

if df_train is not None and "proba" in df_train.columns and "predicted_label" in df_train.columns:
    st.markdown('<p class="section-title">📈 Position du client dans le jeu d\'entraînement</p>', unsafe_allow_html=True)

    df_acc = df_train[df_train["predicted_label"] == 0]
    df_rej = df_train[df_train["predicted_label"] == 1]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=df_acc["proba"], name="Acceptés (train)", marker_color=COLOR_ACCEPT,
        opacity=0.55, nbinsx=50,
        hovertemplate="Proba : %{x:.2f}<br>Nb : %{y}<extra>Acceptés</extra>",
    ))
    fig_hist.add_trace(go.Histogram(
        x=df_rej["proba"], name="Rejetés (train)", marker_color=COLOR_REJECT,
        opacity=0.55, nbinsx=50,
        hovertemplate="Proba : %{x:.2f}<br>Nb : %{y}<extra>Rejetés</extra>",
    ))
    fig_hist.add_vline(
        x=proba, line_color=COLOR_HIGHLIGHT, line_width=2, line_dash="dash",
        annotation_text=f"  Client #{int(selected_id)}",
        annotation_font=dict(color=COLOR_HIGHLIGHT, size=11),
        annotation_position="top right",
    )
    fig_hist.add_vline(
        x=threshold, line_color="#6b7280", line_width=1, line_dash="dot",
        annotation_text=f"  Seuil {threshold}",
        annotation_font=dict(color="#6b7280", size=10),
        annotation_position="top left",
    )
    fig_hist.update_layout(
        **PLOTLY_LAYOUT, margin=DEFAULT_MARGIN,
        xaxis=DEFAULT_XAXIS, yaxis=DEFAULT_YAXIS,
        barmode="overlay", height=420,
        title=dict(text=f"Distribution des probabilités — {len(df_train):,} clients (entraînement)",
                   font=dict(size=12, color="#9ca3af")),
        xaxis_title="Probabilité de défaut", yaxis_title="Nombre de clients",
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#9ca3af"), x=0.75, y=0.95),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Scatter plot sur les 2 top SHAP features ──
    train_shap_cols = [c for c in df_train.columns if c.startswith("shap_")]
    if shap_values and len(train_shap_cols) >= 2:
        top2 = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:2]
        feat1_shap, feat2_shap = top2[0][0], top2[1][0]
        feat1_raw, feat2_raw = feature_name(feat1_shap), feature_name(feat2_shap)

        if feat1_shap in df_train.columns and feat2_shap in df_train.columns:
            st.markdown('<p class="section-title">🔬 Scatter — Top 2 features SHAP (vs entraînement)</p>', unsafe_allow_html=True)

            fig_scatter = go.Figure()

            for lbl, sub, color in [("Acceptés", df_acc, COLOR_ACCEPT), ("Rejetés", df_rej, COLOR_REJECT)]:
                fig_scatter.add_trace(go.Scattergl(
                    x=sub[feat1_shap], y=sub[feat2_shap],
                    mode="markers", name=f"{lbl} (train)",
                    marker=dict(color=color, size=3, opacity=0.3),
                    hovertemplate=f"{feature_label(feat1_shap)} : %{{x:.4f}}<br>{feature_label(feat2_shap)} : %{{y:.4f}}<extra>{lbl}</extra>",
                ))

            client_x = shap_values.get(feat1_shap, 0)
            client_y = shap_values.get(feat2_shap, 0)
            fig_scatter.add_trace(go.Scatter(
                x=[client_x], y=[client_y],
                mode="markers+text", name=f"Client #{int(selected_id)}",
                marker=dict(color=COLOR_HIGHLIGHT, size=14, symbol="diamond", line=dict(width=2, color="#fff")),
                text=[f"#{int(selected_id)}"], textposition="top center",
                textfont=dict(color=COLOR_HIGHLIGHT, size=11),
            ))

            fig_scatter.update_layout(
                **PLOTLY_LAYOUT, margin=DEFAULT_MARGIN,
                xaxis={**DEFAULT_XAXIS, "title": feature_label(feat1_shap)},
                yaxis={**DEFAULT_YAXIS, "title": feature_label(feat2_shap)},
                height=480,
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color="#9ca3af")),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.markdown(HTML_TRAIN_MISSING, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Row 4 — Tableau récap SHAP
# ─────────────────────────────────────────────

if shap_values:
    st.markdown('<p class="section-title">📋 Détail des valeurs SHAP du client</p>', unsafe_allow_html=True)

    recap = pd.DataFrame({
        "Feature": [f"{feature_label(c)}  [{feature_name(c)}]" for c in shap_values.keys()],
        "Valeur SHAP": [round(v, 4) for v in shap_values.values()],
        "Impact": ["⬆ Défaut" if v > 0 else "⬇ Non-défaut" for v in shap_values.values()],
    }).sort_values("Valeur SHAP", key=abs, ascending=False).reset_index(drop=True)

    st.dataframe(
        recap,
        use_container_width=True,
        hide_index=True,
        column_config={"Valeur SHAP": st.column_config.NumberColumn(format="%.4f")},
    )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown(HTML_FOOTER, unsafe_allow_html=True)
