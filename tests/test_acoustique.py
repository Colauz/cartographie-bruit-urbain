"""
Tests de l'arithmétique acoustique.

C'est la partie du projet où une erreur ne se voit pas : une moyenne
arithmétique de décibels produit un nombre parfaitement plausible, simplement
faux, et systématiquement trop bas. Ces tests sont là pour ça.

    python -m pytest tests/ -v
"""

import numpy as np
import pytest

from src.acoustique import (
    lday, lden, ln_nuit, moyenne_energetique, niveau_recepteur,
    puissance_ligne, somme_energetique,
)
from src.indicateurs import SEUIL_PNB_LDEN, calculer, classer


class TestArithmetiqueDecibels:
    def test_deux_sources_identiques_ajoutent_trois_decibels(self):
        """Doubler l'énergie ajoute 3 dB. C'est la règle fondatrice du domaine."""
        assert somme_energetique(np.array([60.0, 60.0])) == pytest.approx(63.01, abs=0.01)

    def test_dix_sources_identiques_ajoutent_dix_decibels(self):
        assert somme_energetique(np.full(10, 50.0)) == pytest.approx(60.0, abs=0.01)

    def test_moyenne_de_niveaux_egaux_est_ce_niveau(self):
        assert moyenne_energetique(np.array([65.0, 65.0, 65.0])) == pytest.approx(65.0)

    def test_la_moyenne_energetique_est_tiree_par_le_haut(self):
        """
        50 et 70 dB donnent 67 dB, pas 60. Une moyenne arithmétique ici
        sous-estimerait la gêne de 7 dB — un facteur 5 en énergie.
        """
        energetique = moyenne_energetique(np.array([50.0, 70.0]))
        arithmetique = np.mean([50.0, 70.0])
        assert energetique == pytest.approx(67.03, abs=0.01)
        assert energetique - arithmetique > 6.9

    def test_source_faible_negligeable(self):
        """Une source 20 dB sous une autre ne change rien : +0,04 dB."""
        assert somme_energetique(np.array([70.0, 50.0])) == pytest.approx(70.04, abs=0.01)


class TestLden:
    def test_niveau_constant_donne_un_lden_superieur(self):
        """
        À 60 dB constant sur 24 h, le Lden vaut environ 66,4 dB : les
        pénalisations de soirée (+5) et de nuit (+10) ne sont pas cosmétiques.
        """
        assert lden(np.full(24, 60.0)) == pytest.approx(66.42, abs=0.05)

    def test_bruit_uniquement_diurne(self):
        """Du bruit de 6 h à 18 h seulement : le Lden reste sous le niveau diurne."""
        profil = np.full(24, 35.0)
        profil[6:18] = 70.0
        assert lden(profil) < 70.0
        assert lday(profil) == pytest.approx(70.0, abs=0.01)

    def test_le_bruit_nocturne_pese_plus_lourd(self):
        """Même énergie totale, déplacée la nuit : le Lden augmente."""
        jour = np.full(24, 40.0)
        jour[8:12] = 70.0
        nuit = np.full(24, 40.0)
        nuit[0:4] = 70.0
        assert lden(nuit) > lden(jour) + 5

    def test_ln_ignore_les_heures_diurnes(self):
        profil = np.full(24, 30.0)
        profil[10] = 90.0   # une pointe en plein jour
        assert ln_nuit(profil) == pytest.approx(30.0, abs=0.01)

    def test_profil_incomplet_rejete(self):
        with pytest.raises(ValueError, match="24 niveaux"):
            lden(np.full(12, 60.0))


class TestModeleEmission:
    def test_doubler_le_debit_ajoute_trois_decibels(self):
        """Deux fois plus de véhicules, deux fois plus d'énergie : +3 dB."""
        un = puissance_ligne(np.array([1000.0]), np.array([0.0]), 50)
        deux = puissance_ligne(np.array([2000.0]), np.array([0.0]), 50)
        assert float((deux - un)[0]) == pytest.approx(3.01, abs=0.02)

    def test_un_poids_lourd_pese_environ_huit_vehicules_legers(self):
        pl = puissance_ligne(np.array([0.0]), np.array([100.0]), 50)
        vl = puissance_ligne(np.array([800.0]), np.array([0.0]), 50)
        assert float((pl - vl)[0]) == pytest.approx(0.0, abs=0.6)

    def test_doubler_la_distance_retire_trois_decibels(self):
        """Source linéique : divergence cylindrique, −3 dB par doublement."""
        puissance = puissance_ligne(np.array([1000.0]), np.array([50.0]), 50)
        proche = niveau_recepteur(puissance, 10)
        loin = niveau_recepteur(puissance, 20)
        assert float((proche - loin)[0]) == pytest.approx(3.01, abs=0.02)

    def test_ordre_de_grandeur_rue_urbaine(self):
        """1 000 véh/h à 50 km/h, à 10 m : entre 60 et 70 dB(A)."""
        puissance = puissance_ligne(np.array([930.0]), np.array([70.0]), 50)
        niveau = float(niveau_recepteur(puissance, 10)[0])
        assert 60 <= niveau <= 70, f"{niveau:.1f} dB(A) hors des ordres de grandeur attendus"

    def test_trafic_nul_ne_produit_pas_moins_l_infini(self):
        puissance = puissance_ligne(np.array([0.0]), np.array([0.0]), 50)
        assert np.isfinite(puissance).all()


class TestClassement:
    @pytest.mark.parametrize("valeur, attendu", [
        (48.0, "< 55 dB"),
        (57.2, "55 – 60 dB"),
        (63.0, "60 – 65 dB"),
        (66.5, "65 – 68 dB"),
        (69.1, "68 – 75 dB"),
        (78.0, "> 75 dB"),
    ])
    def test_classes_d_exposition(self, valeur, attendu):
        assert classer(valeur)[0] == attendu

    def test_seuil_point_noir(self):
        profil = np.full(24, 62.0)   # Lden ≈ 68,4 dB
        indicateurs = calculer(profil, "TEST")
        assert indicateurs.lden >= SEUIL_PNB_LDEN
        assert indicateurs.point_noir

    def test_amplitude_jour_nuit_detecte_la_nuisance_permanente(self):
        """
        Un écart jour/nuit faible signale une nuisance permanente — une usine,
        un axe de transit — et non une pointe de trafic domicile-travail.
        """
        permanent = calculer(np.full(24, 62.0), "PERMANENT")
        pendulaire = np.full(24, 45.0)
        pendulaire[7:9] = 72.0
        pendulaire[17:19] = 72.0
        pointes = calculer(pendulaire, "POINTES")
        assert permanent.amplitude_jour_nuit < 1.0
        assert pointes.amplitude_jour_nuit > permanent.amplitude_jour_nuit
