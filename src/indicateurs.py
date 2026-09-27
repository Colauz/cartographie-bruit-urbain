"""
Classement réglementaire des points de mesure.

Les seuils sont ceux des points noirs du bruit routier (arrêté du 4 avril 2006)
et les classes d'exposition de la directive européenne 2002/49/CE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .acoustique import lday, lden, ln_nuit

# Seuils « point noir du bruit » pour une infrastructure routière.
SEUIL_PNB_LDEN = 68.0
SEUIL_PNB_NUIT = 62.0

# Classes d'exposition Lden de la directive 2002/49/CE, avec une couleur par classe.
CLASSES_LDEN = [
    (0.0, 55.0, "< 55 dB", "#2E7D5B", "exposition faible"),
    (55.0, 60.0, "55 – 60 dB", "#8FBF6A", "exposition modérée"),
    (60.0, 65.0, "60 – 65 dB", "#E4C64F", "exposition notable"),
    (65.0, 68.0, "65 – 68 dB", "#E09B3D", "proche du seuil réglementaire"),
    (68.0, 75.0, "68 – 75 dB", "#C8542F", "point noir du bruit"),
    (75.0, 200.0, "> 75 dB", "#8C2A1B", "exposition critique"),
]


@dataclass
class Indicateurs:
    identifiant: str
    lden: float
    lday: float
    lnuit: float
    laeq_max_horaire: float
    heure_la_plus_bruyante: int
    amplitude_jour_nuit: float
    point_noir: bool
    depassement_nuit: bool
    classe: str
    couleur: str
    commentaire: str


def classer(valeur_lden: float) -> tuple[str, str, str]:
    for borne_basse, borne_haute, libelle, couleur, commentaire in CLASSES_LDEN:
        if borne_basse <= valeur_lden < borne_haute:
            return libelle, couleur, commentaire
    return CLASSES_LDEN[-1][2], CLASSES_LDEN[-1][3], CLASSES_LDEN[-1][4]


def calculer(profil_horaire: np.ndarray, identifiant: str) -> Indicateurs:
    """profil_horaire : 24 niveaux LAeq horaires, en dB(A)."""
    valeur_lden = float(lden(profil_horaire))
    valeur_nuit = float(ln_nuit(profil_horaire))
    valeur_jour = float(lday(profil_horaire))
    libelle, couleur, commentaire = classer(valeur_lden)

    return Indicateurs(
        identifiant=identifiant,
        lden=round(valeur_lden, 1),
        lday=round(valeur_jour, 1),
        lnuit=round(valeur_nuit, 1),
        laeq_max_horaire=round(float(np.max(profil_horaire)), 1),
        heure_la_plus_bruyante=int(np.argmax(profil_horaire)),
        amplitude_jour_nuit=round(valeur_jour - valeur_nuit, 1),
        point_noir=valeur_lden >= SEUIL_PNB_LDEN,
        depassement_nuit=valeur_nuit >= SEUIL_PNB_NUIT,
        classe=libelle,
        couleur=couleur,
        commentaire=commentaire,
    )


def tableau_indicateurs(mesures: pd.DataFrame, capteurs: pd.DataFrame) -> pd.DataFrame:
    """
    mesures : colonnes capteur_id, horodatage, laeq_db
    capteurs : colonnes capteur_id, nom, latitude, longitude, type_voie…

    Le profil horaire moyen est calculé par moyenne énergétique sur les jours
    disponibles, heure par heure — jamais par moyenne arithmétique des décibels.
    """
    travail = mesures.copy()
    travail["horodatage"] = pd.to_datetime(travail["horodatage"])
    travail["heure"] = travail["horodatage"].dt.hour
    travail["pression"] = 10 ** (travail["laeq_db"] / 10)

    profils = (
        travail.groupby(["capteur_id", "heure"])["pression"].mean()
        .unstack("heure")
        .reindex(columns=range(24))
    )
    profils_db = 10 * np.log10(profils)

    lignes = []
    for capteur_id, profil in profils_db.iterrows():
        if profil.isna().any():
            continue
        indicateurs = calculer(profil.to_numpy(), str(capteur_id))
        ligne = indicateurs.__dict__ | {"profil_horaire": [round(v, 1) for v in profil.to_numpy()]}
        lignes.append(ligne)

    table = pd.DataFrame(lignes).rename(columns={"identifiant": "capteur_id"})
    return table.merge(capteurs, on="capteur_id", how="left").sort_values("lden", ascending=False)


def synthese(table: pd.DataFrame) -> dict:
    return {
        "nb_capteurs": int(len(table)),
        "lden_median": round(float(table["lden"].median()), 1),
        "lden_max": round(float(table["lden"].max()), 1),
        "points_noirs": int(table["point_noir"].sum()),
        "depassements_nuit": int(table["depassement_nuit"].sum()),
        "part_au_dessus_de_65": round(float((table["lden"] >= 65).mean()), 3),
        "amplitude_jour_nuit_mediane": round(float(table["amplitude_jour_nuit"].median()), 1),
        "repartition_classes": table["classe"].value_counts().to_dict(),
    }
