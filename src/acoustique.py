"""
Le socle acoustique : arithmétique en décibels et modèle d'émission routière.

Le piège central de ce domaine tient en une phrase : **les décibels ne
s'additionnent pas**. 60 dB + 60 dB font 63 dB, pas 120. Toute moyenne, toute
somme, toute agrégation horaire doit repasser par les pressions acoustiques.
Une moyenne arithmétique de dB est la première erreur de tout traitement de
données de bruit, et elle sous-estime systématiquement les pointes.
"""

from __future__ import annotations

import numpy as np

# --- Paramètres du modèle d'émission -------------------------------------
# Modèle simplifié de type source linéique. Il reproduit les ordres de grandeur
# et les variations relatives, pas la précision d'un calcul réglementaire :
# une carte de bruit officielle s'établit avec CNOSSOS-EU (directive 2015/996)
# ou NMPB-2008, qui intègrent météo, topographie, réflexions et diffraction.

LW_VEHICULE_LEGER_50 = 90.0    # puissance acoustique d'un VL à 50 km/h, dB(A)
PENTE_VITESSE = 30.0           # le bruit de roulement croît en 30·log10(v)
SURPLUS_POIDS_LOURD = 9.0      # un PL vaut environ 8 VL
CORRECTION_SITE_URBAINE = 6.0  # réflexions sur les façades en rue encaissée


def moyenne_energetique(niveaux: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """
    LAeq d'un ensemble de niveaux : moyenne des pressions, pas des décibels.

    >>> round(float(moyenne_energetique(np.array([60.0, 60.0]))), 1)
    60.0
    >>> round(float(moyenne_energetique(np.array([50.0, 70.0]))), 1)
    67.0
    """
    niveaux = np.asarray(niveaux, dtype=np.float64)
    return 10 * np.log10(np.nanmean(10 ** (niveaux / 10), axis=axis))


def somme_energetique(niveaux: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """
    Niveau résultant de sources simultanées.

    >>> round(float(somme_energetique(np.array([60.0, 60.0]))), 1)
    63.0
    """
    niveaux = np.asarray(niveaux, dtype=np.float64)
    return 10 * np.log10(np.nansum(10 ** (niveaux / 10), axis=axis))


def puissance_ligne(debit_vl: np.ndarray, debit_pl: np.ndarray, vitesse: float) -> np.ndarray:
    """
    Puissance acoustique par mètre de voie, dB(A)/m.

    Le débit de poids lourds est converti en débit équivalent de véhicules
    légers avant sommation : deux sources de puissances différentes ne se
    somment pas en comptant les véhicules.
    """
    lw_vl = LW_VEHICULE_LEGER_50 + PENTE_VITESSE * np.log10(max(vitesse, 20) / 50)
    lw_pl = lw_vl + SURPLUS_POIDS_LOURD

    # Densité linéique de véhicules : débit (véh/h) / vitesse (m/h).
    metres_par_heure = max(vitesse, 5) * 1000
    densite_vl = np.maximum(debit_vl, 0) / metres_par_heure
    densite_pl = np.maximum(debit_pl, 0) / metres_par_heure

    puissance = densite_vl * 10 ** (lw_vl / 10) + densite_pl * 10 ** (lw_pl / 10)
    # Un trafic nul ne donne pas -inf : on plancher au bruit de fond résiduel.
    return 10 * np.log10(np.maximum(puissance, 1e-9))


def niveau_recepteur(
    puissance_lineique: np.ndarray,
    distance_m: float,
    correction_site: float = CORRECTION_SITE_URBAINE,
) -> np.ndarray:
    """
    Niveau au point de mesure : divergence cylindrique depuis une source linéique.

    Lp = LW' − 10·log10(d) − 8 + correction de site
    Le terme −8 vient de la divergence hémi-cylindrique, 10·log10(2π) ≈ 8 dB.
    """
    distance = max(distance_m, 1.0)
    return puissance_lineique - 10 * np.log10(distance) - 8 + correction_site


def lden(niveaux_horaires: np.ndarray) -> float:
    """
    Indicateur Lden de la directive 2002/49/CE, sur 24 niveaux horaires.

    Les périodes sensibles sont pénalisées : +5 dB le soir, +10 dB la nuit.
    Ce n'est pas une mesure physique mais un indicateur de gêne — c'est pour
    cette raison qu'un Lden se calcule et ne se mesure pas.

    Découpage français : jour 6h-18h, soirée 18h-22h, nuit 22h-6h.
    """
    niveaux = np.asarray(niveaux_horaires, dtype=np.float64)
    if niveaux.shape[-1] != 24:
        raise ValueError(f"24 niveaux horaires attendus, {niveaux.shape[-1]} reçus")

    jour = niveaux[..., 6:18]
    soiree = niveaux[..., 18:22]
    nuit = np.concatenate([niveaux[..., 22:24], niveaux[..., 0:6]], axis=-1)

    energie = (
        12 * 10 ** (jour / 10).mean(axis=-1)
        + 4 * 10 ** ((soiree + 5) / 10).mean(axis=-1)
        + 8 * 10 ** ((nuit + 10) / 10).mean(axis=-1)
    ) / 24
    return float(10 * np.log10(energie)) if energie.ndim == 0 else 10 * np.log10(energie)


def ln_nuit(niveaux_horaires: np.ndarray) -> float:
    """LAeq de la période nocturne 22h-6h, sans pénalisation."""
    niveaux = np.asarray(niveaux_horaires, dtype=np.float64)
    nuit = np.concatenate([niveaux[..., 22:24], niveaux[..., 0:6]], axis=-1)
    return moyenne_energetique(nuit, axis=-1)


def lday(niveaux_horaires: np.ndarray) -> float:
    niveaux = np.asarray(niveaux_horaires, dtype=np.float64)
    return moyenne_energetique(niveaux[..., 6:18], axis=-1)
