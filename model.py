# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Modèle gains – Acquisition & Confiance", layout="wide")

# ======================
# Sidebar — paramètres
# ======================
st.sidebar.header("Paramètres principaux")
prix = st.sidebar.number_input("Prix du médicament (€ / patient)", min_value=0.0, value=120.0, step=5.0)
nb_med_target = st.sidebar.slider("Médecins cible (atteints à M_acq)", 1, 2000, 50, 1)
pot_diag = st.sidebar.slider("Potentiel max de diagnostics / mois / médecin", 0, 100, 5, 1)
taux_trait_pct = st.sidebar.slider("Taux de patients traités (%)", 0, 100, 40, 1)

st.sidebar.header("Acquisition des médecins")
mois_acq = st.sidebar.slider("M_acq (mois pour atteindre la cible)", 1, 12, 6, 1)
mode_acq = st.sidebar.selectbox("Forme d’acquisition", ["Linéaire", "Logistique (sigmoïde)"])

st.sidebar.header("Confiance des médecins")
mode_conf = st.sidebar.selectbox("Décroissance de la confiance (c_t)", ["Linéaire (c_10 = 0)", "Exponentielle (≈0 à 10)"])
if "Exponentielle" in mode_conf:
    t_half = st.sidebar.slider("Demi-vie de la confiance (mois)", 1, 10, 4, 1)

# ======================
# Horizon temporel
# ======================
# 0 → 12 mois (inclus)
mois = np.arange(0, 13)

# ======================
# Courbe d’acquisition des médecins: med(t)
# ======================
def acquisition_lineaire(t, cible, m_acq):
    # 0 à m_acq : montée linéaire, puis plateau
    ratio = np.clip(t / m_acq, 0, 1)
    return cible * ratio

def acquisition_sigmoide(t, cible, m_acq):
    # Sigmoïde centrée vers m_acq/2 et "quasi-1" à m_acq
    # Choix de k pour avoir s(0)≈0.01 et s(m_acq)≈0.99
    if m_acq <= 0:
        return np.full_like(t, cible, dtype=float)
    k = (2 * np.log(99)) / m_acq
    t0 = m_acq / 2.0
    s = 1 / (1 + np.exp(-k * (t - t0)))
    # Normalisation pour s(0)~0.01 et s(m_acq)~0.99 → déjà assuré par le choix de k/t0
    s = np.clip(s, 0, 1)
    return cible * s

if mode_acq.startswith("Linéaire"):
    med_t = acquisition_lineaire(mois, nb_med_target, mois_acq)
else:
    med_t = acquisition_sigmoide(mois, nb_med_target, mois_acq)

# ======================
# Confiance c(t) (C0 = 1) : nulle à partir de t >= 10
# ======================
c0 = 1.0
if mode_conf.startswith("Linéaire"):
    c_t = c0 * (1 - mois / 10.0)
    c_t = np.clip(c_t, 0, 1)
else:
    c_t = c0 * (0.5 ** (mois / float(t_half)))
    c_t = np.clip(c_t, 0, 1)
    c_t[mois >= 10] = 0.0  # force à 0 à partir du mois 10

# ======================
# Calculs financiers
# ======================
taux_trait = taux_trait_pct / 100.0
# Diagnostics au mois t : med(t) * pot_diag * c(t)
diag_t = med_t * pot_diag * c_t
# Patients traités
traites_t = diag_t * taux_trait
# Revenu
revenu_mensuel = traites_t * prix
revenu_cumule = np.cumsum(revenu_mensuel)

# ======================
# UI – Titre / KPIs
# ======================
st.title("Modèle de gains – Acquisition des médecins & décroissance de la confiance")
st.caption("• Acquisition: 0 → N cible (atteinte à M_acq) • Confiance: 1 → 0 à 10 mois • Horizon: 0–12 mois")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Médecins à 12 mois", f"{int(med_t[-1]):,}".replace(",", " "))
c2.metric("Diagnostics totaux (0–12)", f"{diag_t.sum():,.0f}".replace(",", " "))
c3.metric("Patients traités (0–12)", f"{traites_t.sum():,.0f}".replace(",", " "))
c4.metric("Revenu cumulé (0–12)", f"{revenu_cumule[-1]:,.0f} €".replace(",", " "))

# ======================
# Graphiques
# ======================
tab1, tab2, tab3, tab4 = st.tabs(["Médecins", "Confiance", "Revenu mensuel", "Revenu cumulé"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(mois, med_t, label="Médecins acquis")
    ax1.axvline(mois_acq, linestyle="--", label="M_acq", alpha=0.7)
    ax1.set_title("Acquisition des médecins")
    ax1.set_xlabel("Mois")
    ax1.set_ylabel("Médecins")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(mois, c_t, label="Confiance c(t)")
    ax2.axvline(10, linestyle="--", label="Confiance nulle dès M=10", alpha=0.7)
    ax2.set_title("Décroissance de la confiance (c(t))")
    ax2.set_xlabel("Mois")
    ax2.set_ylabel("Confiance (0–1)")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

with tab3:
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.bar(mois, revenu_mensuel, label="Revenu mensuel", width=0.7)
    ax3.set_title("Revenu mensuel")
    ax3.set_xlabel("Mois")
    ax3.set_ylabel("€")
    ax3.grid(True, axis="y")
    ax3.legend()
    st.pyplot(fig3)

with tab4:
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    ax4.plot(mois, revenu_cumule, label="Revenu cumulé")
    ax4.set_title("Revenu cumulé")
    ax4.set_xlabel("Mois")
    ax4.set_ylabel("€")
    ax4.grid(True)
    ax4.legend()
    st.pyplot(fig4)

# ======================
# Tableau récapitulatif
# ======================
st.subheader("Détails mois par mois")
df = pd.DataFrame({
    "Mois": mois,
    "Médecins acquis": np.round(med_t, 2),
    "Confiance c(t)": np.round(c_t, 3),
    "Diagnostics": np.round(diag_t, 2),
    "Patients traités": np.round(traites_t, 2),
    "Revenu mensuel (€)": np.round(revenu_mensuel, 2),
    "Revenu cumulé (€)": np.round(revenu_cumule, 2),
})
st.dataframe(df, use_container_width=True)

st.info(
    "Lecture : le revenu dépend à la fois de l’acquisition de médecins (qui augmente le volume adressable) "
    "et de la confiance (qui décroit et devient nulle à 10 mois). Après 10 mois, sans action de remotivation, "
    "la confiance restant nulle, le revenu s’aplatit même si de nouveaux médecins sont acquis."
)

