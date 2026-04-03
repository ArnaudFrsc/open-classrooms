"""
Dashboard de prédiction crédit — Home Credit
Charge un fichier _explained.csv (ou predictions.csv avec colonnes shap_*)
et affiche pour chaque client :
  - Décision (Accepté / Rejeté) + probabilité
  - Top-10 SHAP values (waterfall-style)
  - Distribution des probabilités (position du client)
  - Scatter plot sur 2 features clés (position du client)
"""

import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# Config page
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Scoring Crédit · Home Credit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS custom — dark banking aesthetic
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0d0f14;
    color: #e8e6e0;
}

/* Header principal */
.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    letter-spacing: -0.02em;
    color: #f0ede6;
    margin-bottom: 0;
    line-height: 1.1;
}
.main-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #6b7280;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
    margin-bottom: 2rem;
}

/* Verdict cards */
.verdict-card {
    border-radius: 4px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border-left: 4px solid;
}
.verdict-accept {
    background: linear-gradient(135deg, #0a1f14 0%, #0d2b1a 100%);
    border-color: #22c55e;
}
.verdict-reject {
    background: linear-gradient(135deg, #1f0a0a 0%, #2b0d0d 100%);
    border-color: #ef4444;
}
.verdict-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin: 0;
    line-height: 1;
}
.verdict-accept .verdict-label { color: #22c55e; }
.verdict-reject .verdict-label { color: #ef4444; }
.verdict-sub {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-top: 6px;
}

/* Metric boxes */
.metric-box {
    background: #161920;
    border: 1px solid #252830;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #f0ede6;
    line-height: 1;
}
.metric-label {
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6b7280;
    margin-top: 6px;
}

/* Section titles */
.section-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #6b7280;
    border-bottom: 1px solid #252830;
    padding-bottom: 8px;
    margin-bottom: 1rem;
    margin-top: 2rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a0c10;
    border-right: 1px solid #1e2028;
}
section[data-testid="stSidebar"] .main-title {
    font-size: 1.4rem;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #161920;
    border: 1px dashed #2d3040;
    border-radius: 4px;
    padding: 1rem;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2d3040; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#9ca3af", size=11),
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(gridcolor="#1e2028", zerolinecolor="#2d3040"),
    yaxis=dict(gridcolor="#1e2028", zerolinecolor="#2d3040"),
)

COLOR_ACCEPT = "#22c55e"
COLOR_REJECT = "#ef4444"
COLOR_NEUTRAL = "#3b82f6"
COLOR_HIGHLIGHT = "#f59e0b"


def load_data(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def get_shap_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("shap_")]


def feature_name(shap_col: str) -> str:
    """shap_EXT_SOURCE_3 → EXT_SOURCE_3"""
    return re.sub(r"^shap_", "", shap_col)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown('<p class="main-title">🏦 Scoring<br>Crédit</p>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Home Credit · Analyse client</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Charger le fichier de prédictions",
        type=["csv", "xlsx"],
        help="Fichier _explained.csv ou predictions.csv avec colonnes shap_*",
    )

    if uploaded:
        df = load_data(uploaded)
        shap_cols = get_shap_cols(df)

        if "SK_ID_CURR" not in df.columns:
            st.error("❌ Colonne SK_ID_CURR absente.")
            st.stop()
        if "predicted_label" not in df.columns:
            st.error("❌ Colonne predicted_label absente.")
            st.stop()
        if "proba" not in df.columns:
            st.error("❌ Colonne proba absente.")
            st.stop()

        st.markdown("---")
        st.markdown('<p class="metric-label">Fichier chargé</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{len(df):,}</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">clients</p>', unsafe_allow_html=True)

        n_rejected = int(df["predicted_label"].sum())
        n_accepted = len(df) - n_rejected
        st.markdown(f"""
        <div style="margin-top:1rem; font-size:0.75rem; color:#6b7280;">
            ✅ Acceptés : <span style="color:{COLOR_ACCEPT}">{n_accepted:,}</span><br>
            ❌ Rejetés  : <span style="color:{COLOR_REJECT}">{n_rejected:,}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Sélection client
        client_ids = sorted(df["SK_ID_CURR"].unique().tolist())
        selected_id = st.selectbox(
            "🔍 Sélectionner un client",
            options=client_ids,
            format_func=lambda x: f"#{int(x)}",
        )

        # Seuil (affiché mais non modifiable ici — déjà appliqué côté API)
        threshold_display = 0.434
        st.markdown(f"""
        <div style="margin-top:1rem; font-size:0.7rem; color:#6b7280; letter-spacing:0.1em;">
            SEUIL APPLIQUÉ : <span style="color:#f59e0b">{threshold_display}</span>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Main — état initial (pas de fichier)
# ─────────────────────────────────────────────

if not uploaded:
    st.markdown('<h1 class="main-title">Analyse de<br><em>scoring crédit</em></h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Chargez un fichier de prédictions dans la barre latérale</p>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:3rem; color:#4b5563; font-size:0.8rem; line-height:2;">
        Le fichier attendu est le fichier CSV produit par l'API<br>
        <code style="color:#6b7280">/predict/explain</code> — il doit contenir :<br><br>
        &nbsp;· <code>SK_ID_CURR</code> — identifiant client<br>
        &nbsp;· <code>predicted_label</code> — 0 ou 1<br>
        &nbsp;· <code>proba</code> — probabilité de défaut<br>
        &nbsp;· <code>shap_*</code> — valeurs SHAP des top features
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# Données client sélectionné
# ─────────────────────────────────────────────

client = df[df["SK_ID_CURR"] == selected_id].iloc[0]
label = int(client["predicted_label"])
proba = float(client["proba"])
is_accepted = label == 0

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown(f'<h1 class="main-title">Client <em>#{int(selected_id)}</em></h1>', unsafe_allow_html=True)
st.markdown('<p class="main-subtitle">Analyse individuelle · Scoring crédit</p>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Row 1 — Verdict + métriques
# ─────────────────────────────────────────────

col_verdict, col_metrics = st.columns([1.4, 1], gap="large")

with col_verdict:
    verdict_class = "verdict-accept" if is_accepted else "verdict-reject"
    verdict_text = "✅ ACCEPTÉ" if is_accepted else "❌ REJETÉ"
    verdict_desc = "Risque faible — dossier favorable" if is_accepted else "Risque élevé — dossier défavorable"

    st.markdown(f"""
    <div class="verdict-card {verdict_class}">
        <p class="verdict-label">{verdict_text}</p>
        <p class="verdict-sub">{verdict_desc}</p>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    proba_pct = round(proba * 100, 1)
    risk_color = COLOR_REJECT if proba >= 0.434 else COLOR_ACCEPT

    # Jauge probabilité
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
                {"range": [0, 43.4], "color": "#0a1f14"},
                {"range": [43.4, 100], "color": "#1f0a0a"},
            ],
            "threshold": {
                "line": {"color": COLOR_HIGHLIGHT, "width": 2},
                "thickness": 0.8,
                "value": 43.4,
            },
        },
        title={"text": "PROBABILITÉ DE DÉFAUT", "font": {"size": 9, "color": "#6b7280", "family": "DM Mono"}},
    ))
    fig_gauge.update_layout(**PLOTLY_LAYOUT, height=200, margin=dict(l=20, r=20, t=40, b=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

# ─────────────────────────────────────────────
# Row 2 — SHAP waterfall
# ─────────────────────────────────────────────

if shap_cols:
    st.markdown('<p class="section-title">📊 Analyse locale SHAP — contribution des features</p>', unsafe_allow_html=True)

    shap_values = {feature_name(c): float(client[c]) for c in shap_cols}
    shap_sorted = dict(sorted(shap_values.items(), key=lambda x: x[1]))

    features = list(shap_sorted.keys())
    values = list(shap_sorted.values())
    colors = [COLOR_REJECT if v > 0 else COLOR_ACCEPT for v in values]

    fig_shap = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker=dict(
            color=colors,
            opacity=0.85,
            line=dict(width=0),
        ),
        hovertemplate="<b>%{y}</b><br>SHAP : %{x:.4f}<extra></extra>",
    ))

    # Ligne zéro
    fig_shap.add_vline(x=0, line_color="#4b5563", line_width=1)

    fig_shap.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        title=dict(
            text="<span style='font-size:10px;color:#6b7280'>Rouge = pousse vers DÉFAUT · Vert = pousse vers NON-DÉFAUT</span>",
            x=0, font=dict(size=10)
        ),
        xaxis_title="Valeur SHAP",
        yaxis=dict(gridcolor="#1e2028", zerolinecolor="#2d3040", tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_shap, use_container_width=True)
else:
    st.info("Aucune colonne SHAP détectée dans le fichier.")

# ─────────────────────────────────────────────
# Row 3 — Distribution probabilités + Scatter
# ─────────────────────────────────────────────

st.markdown('<p class="section-title">📈 Position du client parmi l\'ensemble des prédictions</p>', unsafe_allow_html=True)

col_hist, col_scatter = st.columns(2, gap="large")

# ── Histogramme distribution probabilités ──
with col_hist:
    fig_hist = go.Figure()

    # Distribution acceptés
    df_acc = df[df["predicted_label"] == 0]
    df_rej = df[df["predicted_label"] == 1]

    fig_hist.add_trace(go.Histogram(
        x=df_acc["proba"],
        name="Acceptés",
        marker_color=COLOR_ACCEPT,
        opacity=0.55,
        nbinsx=40,
        hovertemplate="Proba : %{x:.2f}<br>Nb : %{y}<extra>Acceptés</extra>",
    ))
    fig_hist.add_trace(go.Histogram(
        x=df_rej["proba"],
        name="Rejetés",
        marker_color=COLOR_REJECT,
        opacity=0.55,
        nbinsx=40,
        hovertemplate="Proba : %{x:.2f}<br>Nb : %{y}<extra>Rejetés</extra>",
    ))

    # Position du client
    fig_hist.add_vline(
        x=proba,
        line_color=COLOR_HIGHLIGHT,
        line_width=2,
        line_dash="dash",
        annotation_text=f"  Client #{int(selected_id)}",
        annotation_font=dict(color=COLOR_HIGHLIGHT, size=10),
        annotation_position="top right",
    )

    # Seuil
    fig_hist.add_vline(
        x=0.434,
        line_color="#6b7280",
        line_width=1,
        line_dash="dot",
        annotation_text="  Seuil 0.434",
        annotation_font=dict(color="#6b7280", size=9),
        annotation_position="top left",
    )

    fig_hist.update_layout(
        **PLOTLY_LAYOUT,
        barmode="overlay",
        height=350,
        title=dict(text="Distribution des probabilités de défaut", font=dict(size=11, color="#9ca3af")),
        xaxis_title="Probabilité de défaut",
        yaxis_title="Nombre de clients",
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10, color="#9ca3af"),
            x=0.6, y=0.95,
        ),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Scatter plot top-2 features SHAP ──
with col_scatter:
    if shap_cols and len(shap_cols) >= 2:
        # On prend les 2 features avec le plus grand |shap| moyen → les plus informatives
        mean_abs = {feature_name(c): df[c].abs().mean() for c in shap_cols}
        top2 = sorted(mean_abs, key=mean_abs.get, reverse=True)[:2]
        feat_x, feat_y = top2[0], top2[1]

        # Vérifier que ces features existent dans le df (pas les colonnes shap_ mais les vraies)
        has_feat_x = feat_x in df.columns
        has_feat_y = feat_y in df.columns

        if has_feat_x and has_feat_y:
            fig_scatter = px.scatter(
                df,
                x=feat_x,
                y=feat_y,
                color="predicted_label",
                color_discrete_map={0: COLOR_ACCEPT, 1: COLOR_REJECT},
                opacity=0.35,
                labels={
                    "predicted_label": "Décision",
                    feat_x: feat_x,
                    feat_y: feat_y,
                },
                hover_data={"SK_ID_CURR": True, "proba": ":.3f"},
            )

            # Point du client sélectionné
            fig_scatter.add_trace(go.Scatter(
                x=[client[feat_x]],
                y=[client[feat_y]],
                mode="markers",
                marker=dict(
                    color=COLOR_HIGHLIGHT,
                    size=14,
                    symbol="star",
                    line=dict(color="#fff", width=1),
                ),
                name=f"Client #{int(selected_id)}",
                hovertemplate=f"<b>Client #{int(selected_id)}</b><br>{feat_x}: %{{x:.3f}}<br>{feat_y}: %{{y:.3f}}<extra></extra>",
            ))

            fig_scatter.update_layout(
                **PLOTLY_LAYOUT,
                height=350,
                title=dict(
                    text=f"Position · {feat_x} vs {feat_y}",
                    font=dict(size=11, color="#9ca3af"),
                ),
                legend=dict(
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10, color="#9ca3af"),
                ),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        else:
            # Fallback : scatter proba vs rang
            df_sorted = df.sort_values("proba").reset_index(drop=True)
            client_rank = df_sorted[df_sorted["SK_ID_CURR"] == selected_id].index[0]

            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=df_sorted.index,
                y=df_sorted["proba"],
                mode="markers",
                marker=dict(
                    color=df_sorted["predicted_label"].map({0: COLOR_ACCEPT, 1: COLOR_REJECT}),
                    size=3,
                    opacity=0.4,
                ),
                name="Clients",
                hovertemplate="Rang : %{x}<br>Proba : %{y:.3f}<extra></extra>",
            ))
            fig_scatter.add_trace(go.Scatter(
                x=[client_rank],
                y=[proba],
                mode="markers",
                marker=dict(color=COLOR_HIGHLIGHT, size=12, symbol="star"),
                name=f"Client #{int(selected_id)}",
            ))
            fig_scatter.update_layout(
                **PLOTLY_LAYOUT,
                height=350,
                title=dict(text="Probabilité de défaut · rang dans la population", font=dict(size=11, color="#9ca3af")),
                xaxis_title="Rang (trié par proba)",
                yaxis_title="Probabilité de défaut",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Pas assez de colonnes SHAP pour le scatter.")

# ─────────────────────────────────────────────
# Row 4 — Tableau récap client
# ─────────────────────────────────────────────

st.markdown('<p class="section-title">📋 Détail des valeurs SHAP du client</p>', unsafe_allow_html=True)

if shap_cols:
    recap = pd.DataFrame({
        "Feature": [feature_name(c) for c in shap_cols],
        "Valeur SHAP": [round(float(client[c]), 4) for c in shap_cols],
        "Impact": ["⬆ Défaut" if float(client[c]) > 0 else "⬇ Non-défaut" for c in shap_cols],
    }).sort_values("Valeur SHAP", key=abs, ascending=False).reset_index(drop=True)

    st.dataframe(
        recap,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Valeur SHAP": st.column_config.NumberColumn(format="%.4f"),
        },
    )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────

st.markdown("""
<div style="margin-top:3rem; padding-top:1rem; border-top:1px solid #1e2028;
     font-size:0.65rem; color:#374151; letter-spacing:0.1em; text-align:center;">
    HOME CREDIT SCORING DASHBOARD · SHAP ANALYSIS · LIGHTGBM / XGBOOST
</div>
""", unsafe_allow_html=True)
