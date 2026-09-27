# 05 — Cartographie et analyse du bruit urbain

Chaîne complète : mesures horaires de niveau sonore → indicateurs réglementaires
(Lden, Ln, points noirs du bruit) → carte interactive.

Le jeu de démonstration porte sur un réseau de 20 capteurs sur l'agglomération
de Chambéry, mais le pipeline lit n'importe quel CSV respectant le format
documenté plus bas.

```
mesures.csv ─► contrôle qualité ─► profils horaires ─► Lden / Ln ─► carte
capteurs.csv ─┘                    (moyenne énergétique)   │
                                                    classement réglementaire
```

## Démarrer

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python simuler_capteurs.py      # 20 capteurs × 14 jours = 6 720 mesures horaires
python pipeline.py              # indicateurs + carte
python -m pytest tests/ -v      # 23 tests sur l'arithmétique acoustique

xdg-open sorties/carte_bruit.html
```

## Ce que dit la carte

```
Réseau : 20 capteurs
Lden médian 59.5 dB(A) · maximum 69.2 dB(A)
Points noirs (Lden ≥ 68) : 1
Part des points au-dessus de 65 dB : 15 %

! Voie rapide urbaine (VRU)        69.2   67.6   61.0  68 – 75 dB
  Avenue de Lyon                   66.1   64.5   57.7  65 – 68 dB
  Bassens — route de Lyon          65.8   64.1   57.7  65 – 68 dB
  …
  Parc du Verney                   41.7   42.5   32.7  < 55 dB
```

Quatre couches, parce qu'il y a quatre questions distinctes : exposition Lden,
carte de chaleur à l'échelle du quartier, dépassements nocturnes seuls, et
points noirs du bruit. Chaque point affiche **son profil horaire** dessiné en
SVG dans l'infobulle, avec la période nocturne grisée : c'est ce qui distingue
une pointe de trafic d'une nuisance permanente — et donc ce qui dit quoi faire.

## Le piège du domaine : les décibels ne s'additionnent pas

60 dB + 60 dB font **63 dB**, pas 120. Une moyenne arithmétique de décibels
produit un nombre parfaitement plausible, simplement faux, et systématiquement
trop bas. C'est l'erreur qu'on retrouve dans la plupart des traitements de
données de bruit faits par des non-acousticiens.

Exemple concret : 50 dB et 70 dB.

| Méthode | Résultat |
|---|---|
| Moyenne arithmétique | 60,0 dB — **faux** |
| Moyenne énergétique | 67,0 dB — correct |

Sept décibels d'écart, soit un facteur 5 en énergie acoustique. Toute agrégation
du pipeline — profil horaire moyen, poids de la carte de chaleur, somme de
sources simultanées — repasse par les pressions. Cinq tests unitaires verrouillent
cette propriété, dont celui-ci :

```python
def test_deux_sources_identiques_ajoutent_trois_decibels(self):
    """Doubler l'énergie ajoute 3 dB. C'est la règle fondatrice du domaine."""
    assert somme_energetique(np.array([60.0, 60.0])) == pytest.approx(63.01, abs=0.01)
```

Le même raisonnement s'applique à la carte de chaleur : le poids d'un point est
son **énergie relative**, pas son décibel. Sinon un point à 70 dB pèse à peine
plus qu'un point à 60, alors qu'il émet dix fois plus.

## Lden : un indicateur de gêne, pas une mesure

Le Lden de la directive 2002/49/CE pénalise les périodes sensibles : **+5 dB le
soir** (18 h – 22 h), **+10 dB la nuit** (22 h – 6 h). C'est pour cette raison
qu'un Lden se calcule et ne se mesure pas.

Conséquence contre-intuitive, vérifiée par un test : un niveau **constant** de
60 dB sur 24 heures donne un Lden de **66,4 dB**. Les pénalisations ne sont pas
cosmétiques — elles décident du classement réglementaire d'un point.

Seuils appliqués (points noirs du bruit routier, arrêté du 4 avril 2006) :
**Lden ≥ 68 dB(A)** ou **Ln ≥ 62 dB(A)**.

## L'indicateur que personne ne regarde : l'écart jour/nuit

Le pipeline calcule l'amplitude entre le niveau diurne et le niveau nocturne, et
signale les points où elle est **inférieure à 5 dB avec un Lden ≥ 60**.

Un écart faible ne veut pas dire « pas de problème » : il veut dire que le bruit
est **permanent**. Un axe de transit, une installation technique, une zone
logistique. Deux points peuvent afficher le même Lden et appeler des réponses
opposées — un mur antibruit contre une pointe pendulaire, un arrêté d'exploitation
contre une nuisance continue. Ni le Lden ni le Ln ne le disent seuls.

## Les données de démonstration sont dérivées du trafic, pas tirées au sort

`simuler_capteurs.py` n'échantillonne pas des décibels au hasard : il applique
un modèle d'émission de source linéique à un profil de trafic horaire.

```
LW' = 10·log10( Σ densité_véhicule × 10^(LW/10) )      puissance par mètre de voie
Lp  = LW' − 10·log10(d) − 8 + correction_de_site       divergence cylindrique
```

Ce qui en découle, et qu'un tirage aléatoire ne donnerait pas :

- deux pointes quotidiennes (7–9 h, 16–19 h) en semaine, aucune le week-end ;
- la part de poids lourds **augmente la nuit** — le transit continue quand le
  trafic local s'arrête, ce qui explique pourquoi certaines voies ont un faible
  écart jour/nuit ;
- un parc à 42 dB et une voie rapide à 69 dB **pour une raison physique**
  (débit, vitesse, distance), pas par décret du générateur ;
- doubler la distance retire 3 dB, doubler le débit en ajoute 3.

**Limite assumée** : c'est un modèle d'ordre de grandeur. Une carte de bruit
réglementaire s'établit avec CNOSSOS-EU (directive 2015/996) ou NMPB-2008, qui
intègrent météo, topographie, réflexions et diffraction sur les bâtiments. Les
constantes du modèle sont nommées et documentées en tête de `src/acoustique.py`
plutôt que dissimulées dans les formules.

## Utiliser vos propres données

```bash
python pipeline.py --mesures /mon/export.csv --capteurs /mon/reseau.csv
```

| Fichier | Colonnes obligatoires |
|---|---|
| `mesures.csv` | `capteur_id`, `horodatage` (ISO 8601), `laeq_db` |
| `capteurs.csv` | `capteur_id`, `nom`, `latitude`, `longitude`, `type_voie`, `distance_voie_m` |

Le contrôle qualité écarte les valeurs hors de la plage 20–120 dB(A) — un
sonomètre débranché renvoie des zéros, un micro heurté renvoie 140 dB, et **un
seul point à 140 dB écrase la moyenne énergétique d'une journée entière**. Les
capteurs sans couverture des 24 heures sont exclus du Lden et signalés, plutôt
que de produire un indicateur incomplet qui ressemble à un indicateur complet.

## Origine

Ce projet prolonge un travail de recherche mené au **Center for Environmental
Intelligence de VinUniversity (Hanoï)** sur l'impact du trafic urbain sur la
pollution sonore, combinant capteurs IoT et simulation multi-agents (GAMA).
La version présentée ici est autonome, reproductible, et porte sur un territoire
français.

## Structure

```
simuler_capteurs.py          réseau de démonstration dérivé d'un modèle de trafic
pipeline.py                  contrôle qualité → indicateurs → carte
src/acoustique.py            arithmétique en dB, modèle d'émission, Lden / Ln
src/indicateurs.py           classement réglementaire, seuils, synthèse
src/carte.py                 4 couches folium, infobulles avec profil horaire SVG
tests/test_acoustique.py     23 tests, dont les règles des 3 dB
sorties/carte_bruit.html     la carte interactive
```

---

Projet du portfolio de **Laurian Jamin** — data, automatisation et IA appliquée pour les PME.  
Portfolio : <https://colauz.github.io/Portfolio/> · Malt : <https://www.malt.fr/profile/laurianjamin>
