# ==========================
# Team Surface Area (Convex Hull)
# ==========================
import polars as pl
import numpy as np
from scipy.spatial import ConvexHull

def calc_team_surface_area(df: pl.DataFrame) -> pl.DataFrame:
    resultados = []

    # Garantir que colunas necessárias existem
    if not {"x_norm", "y_norm", "team", "period", "timestamp"}.issubset(df.columns):
        raise ValueError("Faltam colunas obrigatórias: x_norm, y_norm, team, period, timestamp")

    # Iterar por equipa / período / timestamp
    for grupo in df.partition_by(["team", "period", "timestamp"], maintain_order=True):
        pts = grupo.select(["x_norm", "y_norm"]).drop_nulls().to_numpy()

        if pts.shape[0] < 3:
            area = 0.0
        else:
            try:
                hull = ConvexHull(pts)
                area = float(hull.volume)
            except Exception:
                area = 0.0

        resultados.append({
            "team": grupo["team"][0],
            "period": grupo["period"][0],
            "timestamp": grupo["timestamp"][0],
            "team_surface_area": area
        })

    return pl.DataFrame(resultados)
