# ==========================
# eps_zones.py  (corrigido, sem buracos; mesma API)
# ==========================

# ==========================
# 1. Imports
# ==========================
import numpy as np
from shapely.geometry import Polygon as ShpPolygon, Polygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi
from voronoi_global import shadow_points  # shadow_points(points, bounding_box)

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
# 3. Helpers
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

def _wedge_polygon(cx, cy, r0, r1, a0, a1, n=48):
    # polígono anelar entre ângulos [a0,a1] com aproximação por arcos
    if a1 < a0:
        a1 += 2*np.pi
    inner = [_pt(cx, cy, r0, t) for t in np.linspace(a0, a1, n)]
    outer = [_pt(cx, cy, r1, t) for t in np.linspace(a1, a0, n)]
    return ShpPolygon(inner + outer).intersection(PITCH_POLY).buffer(0)

def _norm_ang(a):
    return (a + 2*np.pi) % (2*np.pi)

# ==========================
# 4. Função principal — 2 anéis + Voronoi dinâmico (sem buracos)
# ==========================
def calcular_zonas_eps_fixed(
    df_frame,
    ataque_direita=True,
    r_core=3.5,   # raio interno mínimo (estabilidade)
    r_in=12.0,    # fim do anel interno → zonas 5–8
    r_mid=18.0,   # compat
    r_out=None    # fim do anel externo; se None calcula dinamicamente
):
    """
    Zonas EPS segundo o esquema do Nelson:
      - Anel interno [r_core, r_in] com 4 setores iguais → rótulos [5,6,7,8]
      - Anel externo [r_in, r_out] com 8 setores iguais → rótulos [1,2,9,11,12,10,3,4]
    Desenho “com Voronoi”: para cada setor, intersecta TODAS as células Voronoi reais
    com a máscara do setor e agrega (unary_union). Não há buracos.
    Retorna (polys, (cx,cy)) na ordem [5,6,7,8, 1,2,9,11,12,10,3,4].
    """

    # ==========================
    # 4.1 Seeds e centróide B (equipa em posse → fallback: todos)
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

    # r_out dinâmico (cobre o campo a partir do centróide)
    if r_out is None:
        corners = np.array([[PITCH_X0, PITCH_Y0],
                            [PITCH_X1, PITCH_Y0],
                            [PITCH_X1, PITCH_Y1],
                            [PITCH_X0, PITCH_Y1]], dtype=float)
        r_out = float(np.max(np.linalg.norm(corners - np.array([cx, cy]), axis=1)) + 1.0)

    # ==========================
    # 4.2 Voronoi com shadow + clip ao campo
    # ==========================
    bbox = (PITCH_X0, PITCH_X1, PITCH_Y0, PITCH_Y1)
    vor = Voronoi(shadow_points(seeds_all, bbox))
    cells_coords = _clip_regions_to_pitch(vor, seeds_all, bbox)

    # ==========================
    # 4.3 Máscaras fixas por anel (orientadas pela direção de ataque)
    # ==========================
    rot = 0.0 if ataque_direita else np.pi

    # interno: 4 quadrantes iguais
    inner_labels = [5, 6, 7, 8]
    inner_masks = {}
    for k in range(4):
        a0 = rot - np.pi/4 + k*(np.pi/2)
        a1 = a0 + np.pi/2
        inner_masks[inner_labels[k]] = _wedge_polygon(cx, cy, r_core, r_in, a0, a1)

    # externo: 8 octantes iguais
    outer_labels = [1, 2, 9, 11, 12, 10, 3, 4]
    outer_masks = {}
    for k in range(8):
        a0 = rot - np.pi/8 + k*(np.pi/4)
        a1 = a0 + np.pi/4
        outer_masks[outer_labels[k]] = _wedge_polygon(cx, cy, r_in, r_out, a0, a1)

    # ==========================
    # 4.4 Interseção: setor ∩ (TODAS as células Voronoi)
    # ==========================
    buckets = {z: [] for z in inner_labels + outer_labels}

    # percorre células e adiciona recortes a TODOS os setores relevantes
    # (um recorte por célula por setor) → união final cobre 100% do setor
    for poly_coords in cells_coords:
        if poly_coords is None:
            continue
        try:
            cell = ShpPolygon(poly_coords).buffer(0)
            if cell.is_empty:
                continue
        except Exception:
            continue

        # anel interno
        for lab, mask in inner_masks.items():
            inter = cell.intersection(mask)
            if not inter.is_empty:
                if hasattr(inter, "geoms"):
                    for g in inter.geoms:
                        if not g.is_empty:
                            buckets[lab].append(g.buffer(0))
                else:
                    buckets[lab].append(inter.buffer(0))

        # anel externo
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
# 5. Alias + exports
# ==========================
calcular_zonas_eps_nelson = calcular_zonas_eps_fixed
__all__ = ["calcular_zonas_eps_fixed", "calcular_zonas_eps_nelson"]
