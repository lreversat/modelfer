# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Modèle gains – Acquisition sigmoïde & Confiance linéaire", layout="wide")

# -----------------------
# Helpers
# -----------------------
def sigmoid_acquisition(t, start_t, target_increment, m_acq):
    """
    Acquisition sigmoïde commençant à start_t, sur une durée m_acq,
    allant de 0 à target_increment (ajoutée au stock existant).
    On calibre k pour approx 1% à start_t et 99% à start_t+m_acq.
    """
    if m_acq <= 0 or target_increment <= 0:
        return np.zeros_like(t, dtype=float)
    k = (2 * np.log(99)) / m_acq
    t0 = start_t + m_acq / 2.0
    s = 1 / (1 + np.exp(-k * (t - t0)))
    s = np.clip(s, 0, 1)
    return target_increment * s

def linear_confidence(t, start, length=10):
    """
    Confiance linéaire partant de 1 à 'start', décroissant à 0 à start+length,
    puis 0 après. Avant 'start', 0 par défaut (pour pouvoir combiner par max).
    """
    c = np.zeros_like(t, dtype=float)
    mask = (t >= start) & (t <= start + length)
    c[mask] = 1 - (t[mask] - start) / float(length)
    c[t > start + length] = 0.0
    return np.clip(c, 0, 1)

# -----------------------
# Session state defaults
# -----------------------
if "horizon" not in st.session_state:
    st.session_state.horizon = 12  # 0..12
if "vm_month" not in st.session_state:
    st.session_state.vm_month = None
if "extension_active" not in st.session_state:
    st.session_state.extension_active = False

# -----------------------
# Sidebar — paramètres
# -----------------------
st.sidebar.header("Paramètres principaux")
prix = st.sidebar.number_input("Prix du médicament (€ / patient)", min_value=0.0, value=120.0, step=5.0)
nb_med_target = st.sidebar.slider("Médecins cible (cycle courant)", 1, 2000, 50, 1)
pot_diag = st.sidebar.slider("Potentiel max de diagnostics / mois / médecin", 0, 100, 5, 1)
taux_trait_pct = st.sidebar.slider("Taux de patients traités (%)", 0, 100, 40, 1)
m_acq = st.sidebar.slider("M_acq (mois pour atteindre la cible)", 1, 12, 6, 1)

st.sidebar.divider()
st.sidebar.subheader("Campagne VM (relance de confiance)")
vm_select = st.sidebar.slider("Mois de la campagne VM", 0, st.session_state.horizon, min(6, st.session_state.horizon), 1)
if st.sidebar.button("Lancer une campagne VM"):
    st.session_state.vm_month = int(vm_select)

st.sidebar.divider()
st.sidebar.subheader("Extension +12 mois")
# Paramètres d'extension visibles tout le temps (pour préparer avant clic)
nb_med_new = st.sidebar.slider("Nouveaux médecins à recruter sur l’extension (+12 mois)", 0, 5000, 50, 1)
m_acq_ext = st.sidebar.slider("M_acq extension (mois pour atteindre la nouvelle cible)", 1, 12, 6, 1)
if st.sidebar.button("Ajouter +12 mois & recruter"):
    st.session_state.horizon += 12
    st.session_state.extension_active = True

# -----------------------
# Horizon temporel
# -----------------------
T = st.session_state.horizon
t = np.arange(0, T + 1)

# -----------------------
# Acquisition des médecins (sigmoïde fixe)
#  - Phase 1 : de 0 à nb_med_target en m_acq mois
#  - Phase 2 (si extension) : ajout de nb_med_new, démarrage à t_ext_start
# -----------------------
med_phase1 = sigmoid_acquisition(t, start_t=0, target_increment=nb_med_target, m_acq=m_acq)

if st.session_state.extension_active:
    t_ext_start = 12  # l’extension démarre après le 12e mois initial
    med_phase2 = sigmoid_acquisition(t, start_t=t_ext_start, target_increment=nb_med_new, m_acq=m_acq_ext)
else:
    med_phase2 = np.zeros_like(t, dtype=float)

med_t = med_phase1 + med_phase2  # total médecins au cours du temps

# -----------------------
# Confiance des médecins (linéaire fixe)
#  - Base : 1 → 0 de 0 à 10 mois
#  - Campagne VM : reset à 1 au mois vm_month, puis 1 → 0 sur 10 mois
#  - On prend le max des deux (la relance supplante le résiduel)
# -----------------------
c_base = linear_confidence(t, start=0, length=10)
if st.session_state.vm_month is not None:
    c_vm = linear_confidence(t, start=st.session_state.vm_month, length=10)
    c_t = np.maximum(c_base, c_vm)
else:
    c_t = c_base

# -----------------------
# Calculs
# -----------------------
taux_trait = taux_trait_pct / 100.0
diag_t = med_t * pot_diag * c_t
traites_t = diag_t * taux_trait
revenu_mensuel = traites_t * prix
revenu_cumule = np.cumsum(revenu_mensuel)

# -----------------------
# UI — Titre & KPIs
# -----------------------
st.title("Modèle de gains – Acquisition (sigmoïde) & Confiance (linéaire)")
st.caption("Acquisition des médecins : sigmoïde. Confiance : 1 → 0 en 10 mois. Bouton VM pour relancer la confiance, bouton +12 mois pour prolonger et recruter.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Médecins à T final", f"{int(round(med_t[-1])):,}".replace(",", " "))
c2.metric("Diagnostics totaux (0–T)", f"{diag_t.sum():,.0f}".replace(",", " "))
c3.metric("Patients traités (0–T)", f"{traites_t.sum():,.0f}".replace(",", " "))
c4.metric("Revenu cumulé (0–T)", f"{revenu_cumule[-1]:,.0f} €".replace(",", " "))

# -----------------------
# Graphiques
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs(["Médecins", "Confiance", "Revenu mensuel", "Revenu cumulé"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(t, med_t, label="Médecins acquis")
    ax1.set_title("Acquisition des médecins (sigmoïde)")
    ax1.set_xlabel("Mois")
    ax1.set_ylabel("Médecins")
    if st.session_state.extension_active:
        ax1.axvline(12, linestyle="--", alpha=0.7, label="Début extension (+12m)")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(t, c_t, label="Confiance c(t)")
    ax2.set_title("Confiance (linéaire, reset possible via VM)")
    ax2.set_xlabel("Mois")
    ax2.set_ylabel("Confiance (0–1)")
    ax2.set_ylim(0, 1.05)
    if st.session_state.vm_month is not None:
        ax2.axvline(st.session_state.vm_month, linestyle="--", alpha=0.7, label="Campagne VM")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

with tab3:
    fig3, ax3 = plt.subplots(figsize=(9, 5))
    ax3.bar(t, revenu_mensuel, label="Revenu mensuel", width=0.7)
    ax3.set_title("Revenu mensuel")
    ax3.set_xlabel("Mois")
    ax3.set_ylabel("€")
    ax3.grid(True, axis="y")
    ax3.legend()
    st.pyplot(fig3)

with tab4:
    fig4, ax4 = plt.subplots(figsize=(9, 5))
    ax4.plot(t, revenu_cumule, label="Revenu cumulé")
    ax4.set_title("Revenu cumulé")
    ax4.set_xlabel("Mois")
    ax4.set_ylabel("€")
    ax4.grid(True)
    ax4.legend()
    st.pyplot(fig4)

# -----------------------
# Tableau
# -----------------------
st.subheader("Détails mois par mois")
df = pd.DataFrame({
    "Mois": t,
    "Médecins acquis": np.round(med_t, 2),
    "Confiance c(t)": np.round(c_t, 3),
    "Diagnostics": np.round(diag_t, 2),
    "Patients traités": np.round(traites_t, 2),
    "Revenu mensuel (€)": np.round(revenu_mensuel, 2),
    "Revenu cumulé (€)": np.round(revenu_cumule, 2),
})
st.dataframe(df, use_container_width=True)

# -----------------------
# Notes
# -----------------------
st.info(
    "Campagne VM : remet la confiance à 1 au mois choisi, puis redécroît linéairement en 10 mois. "
    "Extension : prolonge l’horizon de 12 mois et ajoute une nouvelle vague d’acquisition sigmoïde "
    "pour 'Nouveaux médecins à recruter'."
)
