# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Modèle de gains – Confiance médecins", layout="wide")

# --- SIDEBAR: paramètres ---
st.sidebar.header("Paramètres principaux")
prix = st.sidebar.number_input("Prix du médicament (€ / dose)", min_value=0.0, value=120.0, step=5.0)
nb_med = st.sidebar.slider("Nombre de médecins généralistes", 1, 1000, 50, 1)
pot_diag = st.sidebar.slider("Potentiel max de diagnostics / mois / médecin", 0, 100, 5, 1)
taux_trait_pct = st.sidebar.slider("Taux de patients traités (%)", 0, 100, 40, 1)
nb_doses = st.sidebar.slider("Nb de doses par patient", 1, 10, 1, 1)

st.sidebar.header("Confiance des médecins")
c0 = st.sidebar.slider("Confiance initiale C₀ (0–1)", 0.0, 1.0, 1.0, 0.05)
mode_conf = st.sidebar.selectbox("Forme de décroissance", ["Linéaire (c_10 = 0)", "Exponentielle (≈0 à 10 mois)"])

# Paramètre d'exponentielle (optionnel) : demi-vie (en mois)
t_half = None
if "Exponentielle" in mode_conf:
    t_half = st.sidebar.slider("Demi-vie de la confiance (mois)", 1, 10, 4, 1)

# --- Horizon fixe 0..10 mois (confiance nulle au mois 10) ---
mois = np.arange(0, 11)  # 0..10

# --- Confiance c_t ---
if "Linéaire" in mode_conf:
    c_t = c0 * (1 - mois / 10.0)
    c_t = np.clip(c_t, 0, 1)
else:
    # c(t) = C0 * 0.5**(t/t_half), puis on tronque à 0 après 10
    c_t = c0 * (0.5 ** (mois / float(t_half)))
    # force à ~0 au-delà de 10 en coupant à 10 (déjà fait par l'horizon)
    c_t = np.clip(c_t, 0, 1)
    # par cohérence avec l'énoncé "nulle à 10 mois"
    c_t[mois == 10] = 0.0

# --- Calculs ---
taux_trait = taux_trait_pct / 100.0
diag_t = nb_med * pot_diag * c_t
traites_t = diag_t * taux_trait
revenu_mensuel = traites_t * nb_doses * prix
revenu_cumule = np.cumsum(revenu_mensuel)

# --- KPIs ---
st.title("Modèle de gains – Confiance des médecins qui décroît jusqu'à 0 à 10 mois")
st.caption("La confiance module le potentiel de diagnostics. À 10 mois, la confiance est nulle : il faut remotiver les médecins.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Diagnostics totaux (0–10)", f"{diag_t.sum():,.0f}".replace(",", " "))
c2.metric("Patients traités (0–10)", f"{traites_t.sum():,.0f}".replace(",", " "))
c3.metric("Revenu cumulé (0–10)", f"{revenu_cumule[-1]:,.0f} €".replace(",", " "))
c4.metric("Confiance initiale C₀", f"{c0:.2f}")

# --- Graphiques ---
tab1, tab2, tab3 = st.tabs(["Revenu cumulé", "Revenu mensuel", "Confiance"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(mois, revenu_cumule, label="Revenu cumulé (0–10 mois)")
    ax1.set_xlabel("Mois")
    ax1.set_ylabel("€")
    ax1.set_title("Revenu cumulé")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.bar(mois, revenu_mensuel, label="Revenu mensuel", width=0.7)
    ax2.set_xlabel("Mois")
    ax2.set_ylabel("€")
    ax2.set_title("Revenu mensuel")
    ax2.grid(True, axis="y")
    ax2.legend()
    st.pyplot(fig2)

with tab3:
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.plot(mois, c_t, label="Confiance c(t)")
    ax3.set_xlabel("Mois")
    ax3.set_ylabel("Confiance (0–1)")
    ax3.set_title("Évolution de la confiance (devient nulle au mois 10)")
    ax3.set_ylim(0, 1.05)
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)

# --- Tableau des valeurs ---
st.subheader("Détails mois par mois")
df = pd.DataFrame({
    "Mois": mois,
    "Confiance c(t)": c_t,
    "Diagnostics": np.round(diag_t, 2),
    "Patients traités": np.round(traites_t, 2),
    "Revenu mensuel (€)": np.round(revenu_mensuel, 2),
    "Revenu cumulé (€)": np.round(revenu_cumule, 2),
})
st.dataframe(df, use_container_width=True)

st.info(
    "Interprétation : la confiance module directement le volume de diagnostics. "
    "À 10 mois, elle est nulle ; il faut prévoir une action de remotivation pour réalimenter la dynamique."
)
