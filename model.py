# app.py
pip install matplotlib
pip install pandas
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Modèle de gains – carences martiales", layout="wide")

# ---- SIDEBAR : paramètres ----
st.sidebar.header("Paramètres")
prix = st.sidebar.number_input("Prix du médicament (€)", min_value=0.0, max_value=10000.0, value=120.0, step=5.0)
nb_medecins = st.sidebar.slider("Nombre de médecins généralistes", min_value=1, max_value=500, value=50, step=1)
diag_mois_par_med = st.sidebar.slider("Nb de nouveaux diagnostics / mois / médecin", min_value=0, max_value=100, value=5, step=1)
taux_traite = st.sidebar.slider("Taux de patients traités (%)", min_value=0, max_value=100, value=40, step=5)
nb_doses = st.sidebar.slider("Nb de doses par patient", min_value=1, max_value=10, value=1, step=1)
duree = st.sidebar.slider("Durée (mois)", min_value=1, max_value=24, value=6, step=1)

st.sidebar.divider()
st.sidebar.subheader("Afficher aussi les scénarios repères")
show_20 = st.sidebar.checkbox("20 % traités", value=True)
show_40 = st.sidebar.checkbox("40 % traités", value=True)
show_60 = st.sidebar.checkbox("60 % traités", value=True)

# ---- Calculs ----
mois = np.arange(0, duree + 1)  # 0 → durée
# Gain cumulé = médecins × diagnostics/mois × mois × taux_traitement × doses × prix
def gains_cumules(taux_pct: float) -> np.ndarray:
    return nb_medecins * diag_mois_par_med * mois * (taux_pct / 100.0) * nb_doses * prix

# Scénario personnalisé
gains_custom = gains_cumules(taux_traite)

# ---- Mise en page ----
st.title("Modèle de gains – Carences martiales")
st.caption("Gain cumulé en fonction du temps. Ajuste les paramètres dans la barre latérale.")

# KPIs (mois final)
nb_diag_total = nb_medecins * diag_mois_par_med * duree
nb_traites_custom = nb_diag_total * (taux_traite/100)
revenu_total_custom = gains_custom[-1]

c1, c2, c3 = st.columns(3)
c1.metric("Diagnostics (6 mois)", f"{nb_diag_total:,.0f}".replace(",", " "))
c2.metric("Patients traités (custom)", f"{nb_traites_custom:,.0f}".replace(",", " "))
c3.metric("Gain total (custom)", f"{revenu_total_custom:,.0f} €".replace(",", " "))

# ---- Graphique ----
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(mois, gains_custom, label=f"Custom – {taux_traite}% traités")
if show_20: ax.plot(mois, gains_cumules(20), label="Repère – 20%")
if show_40: ax.plot(mois, gains_cumules(40), label="Repère – 40%")
if show_60: ax.plot(mois, gains_cumules(60), label="Repère – 60%")

ax.set_title("Gains cumulés selon le temps")
ax.set_xlabel("Mois")
ax.set_ylabel("Gain cumulé (€)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ---- Tableau récapitulatif (optionnel) ----
st.subheader("Détails (mois par mois)")
import pandas as pd
data = {
    "Mois": mois,
    f"Custom {taux_traite}% (€)": gains_custom
}
if show_20: data["Repère 20% (€)"] = gains_cumules(20)
if show_40: data["Repère 40% (€)"] = gains_cumules(40)
if show_60: data["Repère 60% (€)"] = gains_cumules(60)

df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)
