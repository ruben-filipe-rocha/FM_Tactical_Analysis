import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import Voronoi


def shadow_points(points, bounding_box):
    """
    Adiciona pontos sombra para garantir que o Voronoi fecha dentro do campo.
    """
    xmin, xmax, ymin, ymax = bounding_box
    points = np.array(points)

    left = np.copy(points); left[:, 0] = 2 * xmin - points[:, 0]
    right = np.copy(points); right[:, 0] = 2 * xmax - points[:, 0]
    down = np.copy(points); down[:, 1] = 2 * ymin - points[:, 1]
    up = np.copy(points); up[:, 1] = 2 * ymax - points[:, 1]

    return np.concatenate([points, left, right, down, up])


def ponto_centroide_equipa(df_frame):
    """
    Calcula o centroide médio da equipa com posse (já filtrado).
    """
    coords = df_frame[['x_norm', 'y_norm']].dropna().to_numpy()
    if len(coords) == 0:
        return (60, 40)
    return tuple(coords.mean(axis=0))


def calcular_zonas_eps_fixed(centroide, bounding_box=[0, 120, 0, 80], raio=60, ataque_direita=True):
    """
    Retorna as 12 zonas EPS fixas ao redor do centroide, baseadas em ângulos regulares.
    """
    angulos = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    if not ataque_direita:
        angulos += np.pi

    seeds = [centroide]
    for ang in angulos:
        px = centroide[0] + raio * np.cos(ang)
        py = centroide[1] + raio * np.sin(ang)
        seeds.append([px, py])
    seeds = np.array(seeds)

    spoints = shadow_points(seeds, bounding_box)
    vor = Voronoi(spoints)

    campo = Polygon([
        (bounding_box[0], bounding_box[2]),
        (bounding_box[1], bounding_box[2]),
        (bounding_box[1], bounding_box[3]),
        (bounding_box[0], bounding_box[3])
    ])

    zonas = []
    for i in range(len(seeds)):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        if not region or -1 in region:
            continue
        try:
            poly_coords = [vor.vertices[v] for v in region]
            polygon = Polygon(poly_coords)
            clipped = polygon.intersection(campo)
            if clipped.is_valid and not clipped.is_empty:
                zonas.append(clipped.exterior.coords[:])
        except:
            continue

    return zonas  # Deve ter 13 zonas (1 interior + 12 exteriores), ou 12 se omitirmos o centro


def extract_coords(df):
    return df[["x_norm", "y_norm"]].dropna().values.tolist()

