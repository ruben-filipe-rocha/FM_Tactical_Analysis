# ==========================
# eps_zones.py  (COMPLETO • com fallback de import)
# ==========================

# ==========================
# 1. Imports
# ==========================
import numpy as np
from shapely.geometry import Polygon as ShpPolygon, Polygon, LineString
from shapely.ops import unary_union
from scipy.spatial import Voronoi

# tentar import relativo (quando usado como package) e absoluto (quando corrido a solto)
try:
    from .voronoi_global import shadow_points  # uso normal: from library import *
except Exception:
    try:
        from voronoi_global import shadow_points  # uso direto: from eps_zones import …
    except Exception:
        # fallback simples (mantém API: shadow_points(points, bbox))
        def shadow_points(points, bounding_box):
            # ==========================
            # 1. Shadow 4 direções para fechar Voronoi nos limites do campo
            # ==========================
            x0, x1, y0, y1 = bounding_box
            p = np.asarray(points, dtype=float)
            if p.ndim != 2 or p.shape[1] != 2:
                return p
            left = p.copy();  left[:, 0] = x0 - (p[:, 0] - x0)
            right = p.copy(); right[:, 0] = x1 + (x1 - p[:, 0])
            down = p.copy();  down[:, 1] = y0 - (p[:, 1] - y0)
            up = p.copy();    up[:, 1] = y1 + (y1 - p[:, 1])
            return np.vstack([p, left, right, down, up])

# ==========================
# 2. Parâmetros do campo
# ==========================
PITCH_X0, PITCH_X1 = 0.0, 120.0
PITCH_Y0, PITCH_Y1 = 0.0, 80.0
PITCH_POLY = ShpPolygon([
    (PITCH_X0, PITCH_Y0),
    (PITCH_X1, PITCH_Y0),
    (PITCH_X1, PITCH_Y1),
    (PITCH_X0, PITCH_Y1),
])

# ==========================
# 3. Helpers geométricos
# ==========================
def _clip_regions_to_pitch(vor, seeds, bounding_box):
    # corta regiões Voronoi ao retângulo do campo
    campo = Polygon([
        (bounding_box[0], bounding_box[2]),
        (bounding_box[1], bounding_box[2]),
        (bounding_box[1], bounding_box[3]),
        (bounding_box[0], bounding_box[3]),
    ])
    polys = []
    for i in range(len(seeds)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if not region or -1 in region:
            polys.append(None)
            continue
        try:
            poly_coords = [vor.vertices[v] for v in region]
            polygon = Polygon(poly_coords)
            clipped = polygon.intersection(campo)
            polys.append(list(clipped.exterior.coords[:]) if (clipped.is_valid and not clipped.is_empty) else None)
        except Exception:
            polys.append(None)
    return polys

def _pt(cx, cy, r, ang):
    return (cx + r*np.cos(ang), cy + r*np.sin(ang))

def _wedge_polygon(cx, cy, r0, r1, a0, a1, n=64):
    # polígono anelar entre ângulos [a0,a1]
    if a1 < a0:
        a1 += 2*np.pi
    inner = [_pt(cx, cy, r0, t) for t in np.linspace(a0, a1, n)]
    outer = [_pt(cx, cy, r1, t) for t in np.linspace(a1, a0, n)]
    return ShpPolygon(inner + outer).intersection(PITCH_POLY).buffer(0)

# ==========================
# 4. Zonas EPS (setores fixos + preenchimento Voronoi)
# ==========================
def calcular_zonas_eps_fixed(
    df_frame,
    ataque_direita=True,
    r_core=3.5,   # miolo mínimo
    r_in=12.0,    # fim do anel interno (5–8)
    r_mid=18.0,   # compat
    r_out=None    # fim do anel externo; None -> dinâmico
):
    """
    4 setores de 90° em [r_core,r_in]  -> zonas 5,6,7,8
    8 setores de 45° em [r_in,r_out]   -> zonas 1,2,9,11,12,10,3,4
    Limites angulares fixos; interior de cada setor é Voronoi real recortado.
    Retorna (polys, (cx,cy)) na ordem [5,6,7,8, 1,2,9,11,12,10,3,4]
    """
    # ==========================
    # 4.1 Seeds & centróide B
    # ==========================
    seeds_all = df_frame[['x_norm', 'y_norm']].dropna().to_numpy()
    if seeds_all.shape[0] == 0:
        return [None]*12, (60.0, 40.0)

    if 'teammate' in df_frame.columns:
        posse = df_frame.loc[df_frame['teammate'] == True, ['x_norm', 'y_norm']].dropna().to_numpy()
        if posse.shape[0] == 0:
            posse = seeds_all
    else:
        posse = seeds_all
    cx, cy = posse.mean(axis=0)

    # r_out dinâmico
    if r_out is None:
        corners = np.array([[PITCH_X0, PITCH_Y0],
                            [PITCH_X1, PITCH_Y0],
                            [PITCH_X1, PITCH_Y1],
                            [PITCH_X0, PITCH_Y1]], dtype=float)
        r_out = float(np.max(np.linalg.norm(corners - np.array([cx, cy]), axis=1)) + 1.0)

    # ==========================
    # 4.2 Voronoi real com sombras
    # ==========================
    bbox = (PITCH_X0, PITCH_X1, PITCH_Y0, PITCH_Y1)
    vor = Voronoi(shadow_points(seeds_all, bbox))
    cells_coords = _clip_regions_to_pitch(vor, seeds_all, bbox)

    # ==========================
    # 4.3 Máscaras (orientação pelo ataque)
    # ==========================
    rot = 0.0 if ataque_direita else np.pi

    # interno: 4 quadrantes
    inner_labels = [5, 6, 7, 8]
    inner_masks = {}
    for k in range(4):
        a0 = rot - np.pi/4 + k*(np.pi/2)
        a1 = a0 + np.pi/2
        inner_masks[inner_labels[k]] = _wedge_polygon(cx, cy, r_core, r_in, a0, a1)

    # externo: 8 octantes
    outer_labels = [1, 2, 9, 11, 12, 10, 3, 4]
    outer_masks = {}
    for k in range(8):
        a0 = rot - np.pi/8 + k*(np.pi/4)
        a1 = a0 + np.pi/4
        outer_masks[outer_labels[k]] = _wedge_polygon(cx, cy, r_in, r_out, a0, a1)

    # ==========================
    # 4.4 Recortes Voronoi por setor
    # ==========================
    buckets = {z: [] for z in inner_labels + outer_labels}
    for poly_coords in cells_coords:
        if poly_coords is None:
            continue
        try:
            cell = ShpPolygon(poly_coords).buffer(0)
            if cell.is_empty:
                continue
        except Exception:
            continue

        for lab, mask in inner_masks.items():
            inter = cell.intersection(mask)
            if not inter.is_empty:
                if hasattr(inter, "geoms"):
                    for g in inter.geoms:
                        if not g.is_empty:
                            buckets[lab].append(g.buffer(0))
                else:
                    buckets[lab].append(inter.buffer(0))

        for lab, mask in outer_masks.items():
            inter = cell.intersection(mask)
            if not inter.is_empty:
                if hasattr(inter, "geoms"):
                    for g in inter.geoms:
                        if not g.is_empty:
                            buckets[lab].append(g.buffer(0))
                else:
                    buckets[lab].append(inter.buffer(0))

    # ==========================
    # 4.5 União + ordem final
    # ==========================
    ordered = [5, 6, 7, 8, 1, 2, 9, 11, 12, 10, 3, 4]
    polys = []
    for lab in ordered:
        parts = buckets[lab]
        if not parts:
            polys.append(None)
            continue
        uni = unary_union(parts).buffer(0)
        geom = max(uni.geoms, key=lambda g: g.area) if hasattr(uni, "geoms") else uni
        polys.append(list(geom.exterior.coords) if (not geom.is_empty) else None)

    return polys, (cx, cy)

# ==========================
# 5. EPS (convex hull) + arestas Voronoi dentro do EPS (para plotting)
# ==========================
def calcular_eps_e_voronoi(df_frame):
    # pontos válidos
    pts = df_frame[['x_norm', 'y_norm']].dropna().to_numpy()
    if pts.shape[0] < 3:
        return None, []

    # EPS = convex hull ∩ campo
    try:
        hull = ShpPolygon(pts).convex_hull.intersection(PITCH_POLY).buffer(0)
        if hull.is_empty or hull.geom_type != "Polygon":
            return None, []
    except Exception:
        return None, []

    # Voronoi com shadow e clipping ao EPS
    bbox = (PITCH_X0, PITCH_X1, PITCH_Y0, PITCH_Y1)
    sp = shadow_points(pts, bbox)
    vor = Voronoi(sp)

    segs = []
    for (v1, v2) in vor.ridge_vertices:
        if (v1 == -1) or (v2 == -1):  # <-- FIX: sem parêntesis a mais
            continue
        p1 = vor.vertices[v1]
        p2 = vor.vertices[v2]
        try:
            seg_clip = LineString([p1, p2]).intersection(hull)
        except Exception:
            continue
        if seg_clip.is_empty:
            continue

        if seg_clip.geom_type == "MultiLineString":
            for g in seg_clip.geoms:
                if not g.is_empty:
                    coords = np.asarray(g.coords)
                    if coords.shape[0] >= 2:
                        segs.append(coords)
        elif seg_clip.geom_type == "LineString":
            coords = np.asarray(seg_clip.coords)
            if coords.shape[0] >= 2:
                segs.append(coords)

    hull_coords = list(hull.exterior.coords)
    return hull_coords, segs

# ==========================
# 6. Alias + exports
# ==========================
calcular_zonas_eps_nelson = calcular_zonas_eps_fixed
__all__ = [
    "calcular_zonas_eps_fixed",
    "calcular_zonas_eps_nelson",
    "calcular_eps_e_voronoi",
    "PITCH_X0", "PITCH_X1", "PITCH_Y0", "PITCH_Y1", "PITCH_POLY"
]
