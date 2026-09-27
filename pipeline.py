#!/usr/bin/env python3
"""
Chaîne de traitement : mesures brutes → indicateurs réglementaires → carte.

    python simuler_capteurs.py      # produit un jeu de données de démonstration
    python pipeline.py              # calcule les indicateurs et construit la carte
    python pipeline.py --mesures /mon/fichier.csv --capteurs /mon/reseau.csv

Format attendu
--------------
mesures.csv   capteur_id, horodatage (ISO 8601), laeq_db
capteurs.csv  capteur_id, nom, latitude, longitude, type_voie,
              distance_voie_m [, debit_reference_vh, part_poids_lourds, vitesse_kmh]

Toute source respectant ce format convient — sonomètres, capteurs connectés,
ou un export d'un observatoire du bruit.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.carte import construire_carte
from src.indicateurs import SEUIL_PNB_LDEN, SEUIL_PNB_NUIT, synthese, tableau_indicateurs

RACINE = Path(__file__).parent
logger = logging.getLogger("bruit")


def controler_mesures(mesures: pd.DataFrame) -> pd.DataFrame:
    """
    Filtre les valeurs qui ne peuvent pas être du bruit ambiant mesuré.

    Un sonomètre débranché renvoie des valeurs nulles ; un micro saturé ou
    heurté renvoie des pointes à 140 dB. Les deux faussent une moyenne
    énergétique de façon spectaculaire — un seul point à 140 dB écrase la
    moyenne de toute une journée.
    """
    avant = len(mesures)
    valides = mesures[(mesures["laeq_db"] >= 20) & (mesures["laeq_db"] <= 120)]
    ecartees = avant - len(valides)
    if ecartees:
        logger.warning("%d mesures hors plage 20–120 dB(A) écartées (%.2f %%)",
                       ecartees, 100 * ecartees / avant)

    # Un capteur doit couvrir les 24 heures pour qu'un Lden ait un sens.
    heures = pd.to_datetime(valides["horodatage"]).dt.hour
    couverture = valides.assign(heure=heures).groupby("capteur_id")["heure"].nunique()
    incomplets = couverture[couverture < 24].index.tolist()
    if incomplets:
        logger.warning("Capteurs sans couverture 24 h, exclus du Lden : %s",
                       ", ".join(incomplets))
        valides = valides[~valides["capteur_id"].isin(incomplets)]

    return valides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesures", type=Path, default=RACINE / "data" / "mesures.csv")
    parser.add_argument("--capteurs", type=Path, default=RACINE / "data" / "capteurs.csv")
    parser.add_argument("--sorties", type=Path, default=RACINE / "sorties")
    parser.add_argument("-v", "--verbeux", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbeux else logging.INFO,
                        format="%(levelname)-7s %(message)s")

    for chemin in (args.mesures, args.capteurs):
        if not chemin.exists():
            logger.error("Fichier introuvable : %s — lancez d'abord `python simuler_capteurs.py`", chemin)
            return 1

    mesures = pd.read_csv(args.mesures)
    capteurs = pd.read_csv(args.capteurs)
    logger.info("%d mesures sur %d capteurs", len(mesures), capteurs["capteur_id"].nunique())

    mesures = controler_mesures(mesures)
    table = tableau_indicateurs(mesures, capteurs)
    resume = synthese(table)

    args.sorties.mkdir(parents=True, exist_ok=True)

    export = table.copy()
    export["profil_horaire"] = export["profil_horaire"].map(json.dumps)
    export.to_csv(args.sorties / "indicateurs_par_capteur.csv", index=False, encoding="utf-8-sig")
    (args.sorties / "synthese.json").write_text(
        json.dumps(resume, indent=2, ensure_ascii=False), encoding="utf-8")

    chemin_carte = construire_carte(table, args.sorties / "carte_bruit.html")

    # --- Restitution en console --------------------------------------------
    print(f"\nRéseau : {resume['nb_capteurs']} capteurs")
    print(f"Lden médian {resume['lden_median']} dB(A) · maximum {resume['lden_max']} dB(A)")
    print(f"Points noirs (Lden ≥ {SEUIL_PNB_LDEN:.0f}) : {resume['points_noirs']}")
    print(f"Dépassements nocturnes (Ln ≥ {SEUIL_PNB_NUIT:.0f}) : {resume['depassements_nuit']}")
    print(f"Part des points au-dessus de 65 dB : {resume['part_au_dessus_de_65']:.0%}")

    print("\nClassement par exposition")
    print("─" * 74)
    print(f"  {'Capteur':<30} {'Lden':>6} {'Jour':>6} {'Nuit':>6}  {'Classe':<14}")
    print("─" * 74)
    for _, ligne in table.iterrows():
        marque = "!" if ligne["point_noir"] else (":" if ligne["depassement_nuit"] else " ")
        print(f"{marque} {ligne['nom'][:29]:<30} {ligne['lden']:>6.1f} {ligne['lday']:>6.1f} "
              f"{ligne['lnuit']:>6.1f}  {ligne['classe']:<14}")
    print("─" * 74)
    print("  !  point noir du bruit        :  dépassement nocturne seul")

    # L'écart jour/nuit dit quelque chose qu'aucun indicateur seul ne dit :
    # un écart faible signale une nuisance permanente, pas une pointe de trafic.
    permanents = table[(table["amplitude_jour_nuit"] < 5) & (table["lden"] >= 60)]
    if not permanents.empty:
        print(f"\n{len(permanents)} point{'s' if len(permanents) > 1 else ''} à nuisance permanente "
              f"(écart jour/nuit < 5 dB et Lden ≥ 60) :")
        for _, ligne in permanents.iterrows():
            print(f"  · {ligne['nom']} — écart {ligne['amplitude_jour_nuit']:.1f} dB, "
                  f"nuit {ligne['lnuit']:.1f} dB(A)")

    # Chemins relatifs au projet : la sortie console est capturée pour le portfolio.
    relatif = lambda chemin: chemin.resolve().relative_to(RACINE) if chemin.resolve().is_relative_to(RACINE) else chemin
    print(f"\nCarte     : {relatif(chemin_carte)}")
    print(f"Tableau   : {relatif(args.sorties / 'indicateurs_par_capteur.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
