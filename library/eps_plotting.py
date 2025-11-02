
from matplotlib.patches import Polygon as MplPolygon

def desenhar_area_visivel(row, visible_poly_patch, equipa_esquerda):
    visible_poly_patch.set_visible(False)
    if row.get("actor") and isinstance(row.get("visible_area"), list):
        coords_flat = row["visible_area"]
        if len(coords_flat) >= 6 and len(coords_flat) % 2 == 0:
            coords = list(zip(coords_flat[::2], coords_flat[1::2]))
            if row.get("equipa_actor_frame") != equipa_esquerda:
                coords = [(120 - x, 80 - y) for (x, y) in coords]
            visible_poly_patch.set_xy(coords)
            visible_poly_patch.set_visible(True)
