# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Modèle de gains – Sigmoïde & Confiance inversée", layout="wide")

# -----------------------
# Constantes verrouillées
# -----------------------
CONF_T = 10          # la confiance doit atteindre 0 à 10 mois
CONF_P = 3.0         # exponent 'inversée' (>1 = quasi stable au début, chute forte en fin)
ACQ_LENGTH = 6       # acquisition sigmoïde vers la cible en 6 mois
EXTENSION_MONTHS = 12
COUT_PROJET = 50_000  # coût fixe projet (pour ROI 12 mois)

# -----------------------
# Helpers
# -----------------------
def sigmoid_acquisition(t, start_t, target_increment, length=ACQ_LENGTH):
    """
    Acquisition sigmoïde de 0 à target_increment en 'length' mois, démarrant à start_t.
    Calibrée pour ≈1% à start_t et ≈99% à start_t+length.
    """
    t = np.asarray(t, dtype=float)
    if length <= 0 or target_increment <= 0:
        return np.zeros_like(t, dtype=float)
    k = (2 * np.log(99)) / length
    t0 = start_t + length / 2.0
    s = 1 / (1 + np.exp(-k * (t - t0)))
    s = np.clip(s, 0, 1)
    return target_increment * s

def confidence_inverted_drop(t, T=CONF_T, p=CONF_P):
    """
    Confiance 'inversée' : c(t) = 1 - (t/T)^p pour 0<=t<=T, puis 0.
    - p > 1 : quasi stable au début, chute accélérée vers la fin.
    - c(0)=1, c(T)=0.
    """
    t = np.asarray(t, dtype=float)
    c = np.zeros_like(t, dtype=float)
    mask = (t >= 0) & (t <= T)
    x = t[mask] / float(T)
    c[mask] = 1.0 - (x ** p)
    c[t > T] = 0.0
    return np.clip(c, 0.0, 1.0)

def confidence_with_vm_inverted(t, vm_month=None, T=CONF_T, p=CONF_P):
    """
    Combine la confiance de base 'inversée' et, si vm_month est fixé,
    une relance VM avec le même profil sur 10 mois, via max().
    """
    t = np.asarray(t, dtype=float)
    c_base = confidence_inverted_drop(t, T=T, p=p)
    if vm_month is None:
        return c_base
    tau = np.maximum(0.0, t - vm_month)  # temps relatif depuis VM
    c_vm = confidence_inverted_drop(tau, T=T, p=p)
    return np.maximum(c_base, c_vm)

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
st.sidebar.header("Paramètres")
prix = st.sidebar.number_input("Prix du médicament (€ / patient)", min_value=0.0, value=120.0, step=5.0)
nb_med_target = st.sidebar.slider("Médecins cible (cycle courant)", 1, 2000, 50, 1)
pot_diag = st.sidebar.slider("Potentiel max de diagnostics / mois / médecin", 0, 100, 5, 1)
taux_trait_pct = st.sidebar.slider("Taux de patients traités (%)", 0, 100, 40, 1)

st.sidebar.divider()
st.sidebar.subheader("Campagne VM (relance de confiance, profil 10 mois)")
vm_select = st.sidebar.slider("Mois de la campagne VM", 0, st.session_state.horizon, min(6, st.session_state.horizon), 1)
if st.sidebar.button("Lancer une campagne VM"):
    st.session_state.vm_month = int(vm_select)

st.sidebar.divider()
st.sidebar.subheader("Extension (+12 mois, acquisition 6 mois)")
nb_med_new = st.sidebar.slider("Nouveaux médecins à recruter (extension)", 0, 5000, 50, 1)
if st.sidebar.button("Ajouter +12 mois & recruter"):
    st.session_state.horizon += EXTENSION_MONTHS
    st.session_state.extension_active = True

# -----------------------
# Horizon temporel
# -----------------------
T_end = st.session_state.horizon
t = np.arange(0, T_end + 1)

# -----------------------
# Acquisition des médecins (sigmoïde 6 mois)
#   Phase 1: 0 -> nb_med_target en 6 mois
#   Phase 2 (si extension): +nb_med_new à partir de M=12, en 6 mois
# -----------------------
med_phase1 = sigmoid_acquisition(t, start_t=0, target_increment=nb_med_target, length=ACQ_LENGTH)
if st.session_state.extension_active:
    t_ext_start = 12
    med_phase2 = sigmoid_acquisition(t, start_t=t_ext_start, target_increment=nb_med_new, length=ACQ_LENGTH)
else:
    med_phase2 = np.zeros_like(t, dtype=float)
med_t = med_phase1 + med_phase2

# -----------------------
# Confiance des médecins (inversée 10 mois)
#   Base : 1 - (t/10)^p
#   VM   : reset, même profil sur 10 mois à partir de vm_month
#   Confiance effective = max(base, VM)
# -----------------------
if st.session_state.vm_month is not None:
    c_t = confidence_with_vm_inverted(t, vm_month=st.session_state.vm_month, T=CONF_T, p=CONF_P)
else:
    c_t = confidence_inverted_drop(t, T=CONF_T, p=CONF_P)

# -----------------------
# Calculs
# -----------------------
taux_trait = taux_trait_pct / 100.0
diag_t = med_t * pot_diag * c_t
traites_t = diag_t * taux_trait
revenu_mensuel = traites_t * prix
revenu_cumule = np.cumsum(revenu_mensuel)

# ROI sur 12 mois
revenu_12m = revenu_cumule[min(12, len(revenu_cumule)-1)]
roi = (revenu_12m - COUT_PROJET) / COUT_PROJET if COUT_PROJET > 0 else None

# -----------------------
# UI — Titre & KPIs
# -----------------------
st.title("Modèle de gains – Acquisition (sigmoïde 6m) & Confiance inversée (10m)")
st.caption("Confiance : c(t)=1-(t/10)^p (p=3). Acquisition : sigmoïde 6m. Extension : +12m. ROI calculé sur 12 mois (coût 50 000 €).")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Médecins à T final", f"{int(round(med_t[-1])):,}".replace(",", " "))
c2.metric("Diagnostics totaux (0–T)", f"{diag_t.sum():,.0f}".replace(",", " "))
c3.metric("Patients traités (0–T)", f"{traites_t.sum():,.0f}".replace(",", " "))
c4.metric("Revenu cumulé (0–T)", f"{revenu_cumule[-1]:,.0f} €".replace(",", " "))
if roi is not None:
    c5.metric("ROI (12 mois)", f"{roi*100:.1f} %")

# -----------------------
# Graphiques
# -----------------------
tab1, tab2, tab3, tab4 = st.tabs(["Médecins", "Confiance", "Revenu mensuel", "Revenu cumulé"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(t, med_t, label="Médecins acquis")
    ax1.set_title("Acquisition des médecins (sigmoïde, 6 mois)")
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
    if st.session_state.vm_month is not None:
        ax2.axvline(st.session_state.vm_month, linestyle="--", alpha=0.7, label="Campagne VM")
    ax2.set_title("Confiance (inversée, 10 mois)")
    ax2.set_xlabel("Mois")
    ax2.set_ylabel("Confiance (0–1)")
    ax2.set_ylim(0, 1.05)
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
    ax4.axvline(12, linestyle="--", alpha=0.7, label="Fin période ROI 12m")
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

st.info(
    "La confiance suit c(t)=1-(t/10)^p avec p=3 (plate au début, chute en fin). "
    "La campagne VM réinitialise la confiance pour un nouveau cycle de 10 mois. "
    "Le ROI est calculé sur 12 mois avec un coût projet de 50 000 €."
)
