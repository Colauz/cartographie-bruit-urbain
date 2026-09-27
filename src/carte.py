"""
Rendu cartographique interactif (folium / Leaflet).

Trois couches, parce que trois questions différentes :
  · Exposition Lden   quels points dépassent le seuil réglementaire ?
  · Carte de chaleur  où se concentre la gêne à l'échelle du quartier ?
  · Nuit (Ln)         qui subit du bruit quand les gens dorment ?

Chaque point porte son profil horaire dessiné en SVG dans l'infobulle : c'est
ce qui distingue un pic de trafic d'une nuisance permanente, et c'est la seule
information qui dise quoi faire.
"""

from __future__ import annotations

import json
from pathlib import Path

import branca
import folium
import pandas as pd
from folium.plugins import HeatMap, MarkerCluster

from .indicateurs import CLASSES_LDEN, SEUIL_PNB_LDEN, SEUIL_PNB_NUIT

CENTRE_CHAMBERY = (45.5700, 5.9180)

# OpenStreetMap : pas de clé d'API à gérer, pas de dépendance à un fournisseur
# commercial. Les fonds CartoDB, plus sobres, exigent désormais une clé.
FOND_PAR_DEFAUT = "OpenStreetMap"


def _sparkline_svg(profil: list[float], largeur: int = 240, hauteur: int = 58) -> str:
    """Profil horaire dessiné à la main : pas de dépendance graphique côté client."""
    mini, maxi = min(profil), max(profil)
    etendue = max(maxi - mini, 1.0)
    marge = 4

    points = []
    for heure, valeur in enumerate(profil):
        x = marge + heure * (largeur - 2 * marge) / 23
        y = hauteur - marge - (valeur - mini) / etendue * (hauteur - 2 * marge)
        points.append(f"{x:.1f},{y:.1f}")

    # Bande nocturne 22h-6h, pour situer visuellement la période pénalisée.
    x_22h = marge + 22 * (largeur - 2 * marge) / 23
    x_6h = marge + 6 * (largeur - 2 * marge) / 23

    return (
        f'<svg width="{largeur}" height="{hauteur}" viewBox="0 0 {largeur} {hauteur}" '
        f'role="img" aria-label="Profil horaire du niveau sonore">'
        f'<rect x="{marge}" y="0" width="{x_6h - marge:.1f}" height="{hauteur}" fill="#2B3A55" opacity="0.10"/>'
        f'<rect x="{x_22h:.1f}" y="0" width="{largeur - marge - x_22h:.1f}" height="{hauteur}" '
        f'fill="#2B3A55" opacity="0.10"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#1F5C8B" stroke-width="1.8" '
        f'stroke-linejoin="round"/>'
        f'<text x="{marge}" y="{hauteur - 1}" font-size="8" fill="#6B7684">0h</text>'
        f'<text x="{largeur / 2 - 6:.0f}" y="{hauteur - 1}" font-size="8" fill="#6B7684">12h</text>'
        f'<text x="{largeur - marge - 14:.0f}" y="{hauteur - 1}" font-size="8" fill="#6B7684">23h</text>'
        f'</svg>'
    )


def _infobulle(ligne: pd.Series) -> str:
    profil = ligne["profil_horaire"]
    if isinstance(profil, str):
        profil = json.loads(profil)

    alertes = []
    if ligne["point_noir"]:
        alertes.append(f"Lden ≥ {SEUIL_PNB_LDEN:.0f} dB — point noir du bruit")
    if ligne["depassement_nuit"]:
        alertes.append(f"Ln ≥ {SEUIL_PNB_NUIT:.0f} dB — dépassement nocturne")

    bandeau = ""
    if alertes:
        bandeau = (
            '<div style="background:#FBEAE5;border-left:3px solid #C8542F;padding:6px 8px;'
            'margin:8px 0;font-size:11px;color:#8C2A1B">'
            + "<br>".join(alertes) + "</div>"
        )

    return f"""
<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;width:262px">
  <div style="font-weight:600;font-size:13px;color:#1B2430">{ligne['nom']}</div>
  <div style="font-size:10.5px;color:#6B7684;margin-bottom:8px">
    {ligne['capteur_id']} · {ligne['type_voie']} · {ligne['distance_voie_m']:.0f} m de la voie
  </div>

  <div style="display:flex;gap:14px;margin-bottom:8px">
    <div><div style="font-size:9px;color:#6B7684;text-transform:uppercase">Lden</div>
         <div style="font-size:17px;font-weight:600;color:{ligne['couleur']}">{ligne['lden']:.1f}</div></div>
    <div><div style="font-size:9px;color:#6B7684;text-transform:uppercase">Jour</div>
         <div style="font-size:17px;font-weight:600;color:#1B2430">{ligne['lday']:.1f}</div></div>
    <div><div style="font-size:9px;color:#6B7684;text-transform:uppercase">Nuit</div>
         <div style="font-size:17px;font-weight:600;color:#1B2430">{ligne['lnuit']:.1f}</div></div>
  </div>

  {_sparkline_svg(profil)}

  <div style="font-size:10.5px;color:#6B7684;margin-top:6px">
    Pointe à {ligne['heure_la_plus_bruyante']:.0f} h ({ligne['laeq_max_horaire']:.1f} dB) ·
    écart jour/nuit {ligne['amplitude_jour_nuit']:.1f} dB
  </div>
  {bandeau}
  <div style="font-size:10.5px;color:#1B2430;margin-top:4px">{ligne['commentaire'].capitalize()}</div>
</div>
"""


def _legende() -> branca.element.MacroElement:
    entrees = "".join(
        f'<div style="display:flex;align-items:center;gap:7px;margin:2px 0">'
        f'<span style="width:13px;height:13px;border-radius:3px;background:{couleur};'
        f'display:inline-block"></span>'
        f'<span style="font-size:11px;color:#1B2430">{libelle}</span></div>'
        for _, _, libelle, couleur, _ in CLASSES_LDEN
    )
    html = f"""
<div style="position:fixed;bottom:22px;left:14px;z-index:9999;background:rgba(255,255,255,.95);
            padding:11px 13px;border-radius:7px;box-shadow:0 1px 10px rgba(0,0,0,.18);
            font-family:system-ui,-apple-system,sans-serif">
  <div style="font-size:11px;font-weight:600;color:#1B2430;margin-bottom:6px">
    Exposition Lden
  </div>
  {entrees}
  <div style="font-size:9.5px;color:#6B7684;margin-top:7px;max-width:170px;line-height:1.35">
    Seuil point noir du bruit routier : 68 dB(A) en Lden, 62 dB(A) la nuit.
  </div>
</div>
"""
    element = branca.element.MacroElement()
    element._template = branca.element.Template(
        "{% macro html(this, kwargs) %}" + html + "{% endmacro %}"
    )
    return element


def _rayon(valeur_lden: float) -> float:
    """Le rayon suit l'énergie, pas le décibel : 3 dB de plus doublent la surface."""
    return 6 + max(valeur_lden - 45, 0) * 0.65


def construire_carte(table: pd.DataFrame, chemin: Path, centre=CENTRE_CHAMBERY) -> Path:
    carte = folium.Map(location=centre, zoom_start=13, tiles=FOND_PAR_DEFAUT, control_scale=True)

    # --- Couche 1 : exposition Lden ---------------------------------------
    couche_lden = folium.FeatureGroup(name="Exposition Lden", show=True)
    for _, ligne in table.iterrows():
        folium.CircleMarker(
            location=(ligne["latitude"], ligne["longitude"]),
            radius=_rayon(ligne["lden"]),
            color="#FFFFFF",
            weight=1.5,
            fill=True,
            fill_color=ligne["couleur"],
            fill_opacity=0.82,
            popup=folium.Popup(_infobulle(ligne), max_width=300),
            tooltip=f"{ligne['nom']} — {ligne['lden']:.1f} dB(A)",
        ).add_to(couche_lden)
    couche_lden.add_to(carte)

    # --- Couche 2 : carte de chaleur --------------------------------------
    # Le poids est l'énergie acoustique relative, pas le décibel : sinon un
    # point à 70 dB pèse à peine plus qu'un point à 60, alors qu'il émet 10 fois plus.
    reference = table["lden"].min()
    poids = [(r["latitude"], r["longitude"], float(10 ** ((r["lden"] - reference) / 10)))
             for _, r in table.iterrows()]
    maximum = max(p[2] for p in poids)
    poids = [(lat, lon, p / maximum) for lat, lon, p in poids]

    HeatMap(
        poids, name="Carte de chaleur (énergie)", radius=34, blur=26, min_opacity=0.25,
        gradient={0.0: "#2E7D5B", 0.35: "#E4C64F", 0.6: "#E09B3D", 0.85: "#C8542F", 1.0: "#8C2A1B"},
        show=False,
    ).add_to(carte)

    # --- Couche 3 : dépassements nocturnes --------------------------------
    couche_nuit = folium.FeatureGroup(name=f"Dépassements nocturnes (Ln ≥ {SEUIL_PNB_NUIT:.0f} dB)", show=False)
    for _, ligne in table[table["depassement_nuit"]].iterrows():
        folium.CircleMarker(
            location=(ligne["latitude"], ligne["longitude"]),
            radius=_rayon(ligne["lnuit"]),
            color="#8C2A1B", weight=2, fill=True, fill_color="#C8542F", fill_opacity=0.65,
            popup=folium.Popup(_infobulle(ligne), max_width=300),
            tooltip=f"{ligne['nom']} — Ln {ligne['lnuit']:.1f} dB(A)",
        ).add_to(couche_nuit)
    couche_nuit.add_to(carte)

    # --- Couche 4 : points noirs, regroupés -------------------------------
    points_noirs = table[table["point_noir"]]
    if not points_noirs.empty:
        grappe = MarkerCluster(name="Points noirs du bruit", show=False)
        for _, ligne in points_noirs.iterrows():
            folium.Marker(
                location=(ligne["latitude"], ligne["longitude"]),
                icon=folium.Icon(color="darkred", icon="volume-up", prefix="fa"),
                popup=folium.Popup(_infobulle(ligne), max_width=300),
                tooltip=f"Point noir — {ligne['nom']}",
            ).add_to(grappe)
        grappe.add_to(carte)

    folium.LayerControl(collapsed=False).add_to(carte)
    carte.get_root().add_child(_legende())

    chemin.parent.mkdir(parents=True, exist_ok=True)
    carte.save(str(chemin))
    return chemin
