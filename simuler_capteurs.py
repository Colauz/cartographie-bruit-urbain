#!/usr/bin/env python3
"""
Simule un réseau de capteurs de bruit sur l'agglomération de Chambéry.

Le pipeline d'analyse lit un CSV de mesures au format documenté dans le README.
Ce script en produit un jeu, à partir d'un modèle d'émission routière plutôt
que d'un tirage aléatoire : les niveaux sont **dérivés du trafic**, ce qui fait
que le bruit du mardi 8 h ressemble à celui du mercredi 8 h et que la nuit est
calme pour une raison.

    python simuler_capteurs.py --jours 14

Sortie :
    data/capteurs.csv   le réseau : position, type de voie, trafic de référence
    data/mesures.csv    un LAeq par capteur et par heure
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.acoustique import niveau_recepteur, puissance_ligne

DOSSIER_DATA = Path(__file__).parent / "data"

# Réseau fictif mais géographiquement cohérent : les coordonnées tombent sur de
# vraies voies de l'agglomération chambérienne.
CAPTEURS = [
    # nom,                          lat,      lon,     type_voie,   débit jour (véh/h), % PL, vitesse, distance
    ("Avenue des Ducs de Savoie",   45.5712, 5.9204, "artère",         1450, 0.07, 50, 12),
    ("Quai des Allobroges",         45.5698, 5.9165, "artère",         1180, 0.05, 50, 10),
    ("Place du Palais de Justice",  45.5662, 5.9186, "centre-ville",    520, 0.03, 30,  8),
    ("Rue de Boigne",               45.5674, 5.9203, "piétonnier",       60, 0.01, 20, 14),
    ("Faubourg Montmélian",         45.5601, 5.9285, "artère",         1320, 0.09, 50, 11),
    ("Avenue de Lyon",              45.5738, 5.9108, "artère",         1610, 0.08, 50,  9),
    ("Voie rapide urbaine (VRU)",   45.5789, 5.9032, "voie rapide",    3400, 0.12, 90, 35),
    ("Rue Nicolas Parent",          45.5655, 5.9241, "desserte",        310, 0.02, 30,  7),
    ("Hôpital — rue Sainte-Rose",   45.5723, 5.9296, "desserte",        280, 0.04, 30, 18),
    ("Lycée Vaugelas",              45.5688, 5.9152, "centre-ville",    640, 0.03, 30, 15),
    ("Chambéry-le-Haut",            45.5921, 5.9098, "artère",          870, 0.06, 50, 13),
    ("Bassens — route de Lyon",     45.5851, 5.9075, "artère",         1240, 0.11, 70, 16),
    ("La Ravoire — ZAC",            45.5613, 5.9508, "artère",          960, 0.14, 50, 22),
    ("Parc du Verney",              45.5697, 5.9248, "parc",             90, 0.01, 20, 45),
    ("Gare SNCF — parvis",          45.5714, 5.9182, "centre-ville",    710, 0.04, 30, 12),
    ("Rue de la République",        45.5668, 5.9218, "centre-ville",    480, 0.02, 30,  9),
    ("Cognin — avenue de Chambéry", 45.5589, 5.8963, "artère",          890, 0.07, 50, 14),
    ("Jacob-Bellecombette",         45.5577, 5.9101, "desserte",        340, 0.03, 30, 11),
    ("Sonnaz — RD991",              45.6088, 5.8967, "route",           620, 0.16, 80, 25),
    ("Barberaz — collège",          45.5541, 5.9312, "desserte",        290, 0.03, 30, 16),
]

# Profil horaire du trafic, en part du débit de référence.
# Deux pointes : domicile-travail du matin et du soir.
PROFIL_SEMAINE = np.array([
    0.09, 0.05, 0.04, 0.04, 0.07, 0.19, 0.48, 0.92, 1.00, 0.78, 0.71, 0.76,
    0.83, 0.79, 0.74, 0.81, 0.96, 1.00, 0.88, 0.61, 0.44, 0.33, 0.24, 0.15,
])
# Le week-end : pas de pointe du matin, activité étalée et décalée vers le soir.
PROFIL_WEEKEND = np.array([
    0.21, 0.15, 0.10, 0.07, 0.06, 0.08, 0.14, 0.24, 0.38, 0.56, 0.72, 0.83,
    0.86, 0.84, 0.82, 0.83, 0.85, 0.82, 0.74, 0.62, 0.52, 0.44, 0.37, 0.29,
])

# Bruit de fond résiduel par type de tissu urbain, dB(A).
BRUIT_DE_FOND = {
    "voie rapide": 42.0, "artère": 40.0, "route": 36.0,
    "centre-ville": 41.0, "desserte": 35.0, "piétonnier": 38.0, "parc": 31.0,
}

# Sources non routières : terrasses le soir en centre-ville, trains en gare…
EVENEMENTS = {
    "centre-ville": {"heures": range(19, 24), "niveau": 56.0, "probabilite": 0.55},
    "piétonnier": {"heures": range(11, 23), "niveau": 58.0, "probabilite": 0.70},
    "parc": {"heures": range(12, 19), "niveau": 48.0, "probabilite": 0.40},
}


def construire_capteurs() -> pd.DataFrame:
    lignes = []
    for i, (nom, lat, lon, type_voie, debit, part_pl, vitesse, distance) in enumerate(CAPTEURS, start=1):
        lignes.append({
            "capteur_id": f"CH{i:03d}",
            "nom": nom,
            "latitude": lat,
            "longitude": lon,
            "type_voie": type_voie,
            "debit_reference_vh": debit,
            "part_poids_lourds": part_pl,
            "vitesse_kmh": vitesse,
            "distance_voie_m": distance,
        })
    return pd.DataFrame(lignes)


def simuler(capteurs: pd.DataFrame, debut: date, jours: int, graine: int) -> pd.DataFrame:
    alea = np.random.default_rng(graine)
    lignes = []

    for _, capteur in capteurs.iterrows():
        fond = BRUIT_DE_FOND.get(capteur["type_voie"], 38.0)
        evenement = EVENEMENTS.get(capteur["type_voie"])

        for jour_offset in range(jours):
            jour = debut + timedelta(days=jour_offset)
            weekend = jour.weekday() >= 5
            profil = PROFIL_WEEKEND if weekend else PROFIL_SEMAINE

            # Variabilité d'un jour à l'autre : météo, vacances, événements.
            facteur_jour = alea.normal(1.0, 0.08)

            for heure in range(24):
                debit = capteur["debit_reference_vh"] * profil[heure] * facteur_jour
                debit *= alea.normal(1.0, 0.10)          # aléa de comptage horaire
                debit = max(debit, 0.0)

                debit_pl = debit * capteur["part_poids_lourds"]
                # La nuit, la part de poids lourds augmente : le transit continue.
                if heure < 6 or heure >= 22:
                    debit_pl *= 1.8
                debit_vl = max(debit - debit_pl, 0.0)

                puissance = puissance_ligne(
                    np.array([debit_vl]), np.array([debit_pl]), capteur["vitesse_kmh"]
                )
                niveau = float(niveau_recepteur(puissance, capteur["distance_voie_m"])[0])

                # Sources simultanées : route + bruit de fond (+ événement).
                sources = [10 ** (niveau / 10), 10 ** (fond / 10)]
                if evenement and heure in evenement["heures"] and alea.random() < evenement["probabilite"]:
                    sources.append(10 ** (evenement["niveau"] / 10))
                niveau = 10 * np.log10(sum(sources))

                # Incertitude de mesure d'un sonomètre de classe 2.
                niveau += alea.normal(0.0, 0.8)

                lignes.append({
                    "capteur_id": capteur["capteur_id"],
                    "horodatage": datetime.combine(jour, datetime.min.time()) + timedelta(hours=heure),
                    "laeq_db": round(niveau, 1),
                })

    return pd.DataFrame(lignes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jours", type=int, default=14)
    parser.add_argument("--debut", type=date.fromisoformat, default=date(2026, 9, 1))
    parser.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()

    DOSSIER_DATA.mkdir(exist_ok=True)
    capteurs = construire_capteurs()
    mesures = simuler(capteurs, args.debut, args.jours, args.seed)

    capteurs.to_csv(DOSSIER_DATA / "capteurs.csv", index=False, encoding="utf-8")
    mesures.to_csv(DOSSIER_DATA / "mesures.csv", index=False, encoding="utf-8")

    print(f"{len(capteurs)} capteurs · {len(mesures):,} mesures horaires "
          f"sur {args.jours} jours".replace(",", " "))
    print(f"LAeq observé : {mesures['laeq_db'].min():.1f} à {mesures['laeq_db'].max():.1f} dB(A)")
    print(f"\n  {DOSSIER_DATA / 'capteurs.csv'}")
    print(f"  {DOSSIER_DATA / 'mesures.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
