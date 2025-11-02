# ==========================
# Surface Area Ratio (SAR)
# ==========================
import polars as pl
import numpy as np
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

def calc_surface_area_ratio(df: pl.DataFrame) -> pl.DataFrame:
    resultados = []

    # garantir colunas obrigatórias
    if not {"equipa_jogador", "period", "timestamp", "x_norm", "y_norm"}.issubset(df.columns):
        raise ValueError("Faltam colunas obrigatórias: equipa_jogador, period, timestamp, x_norm, y_norm")

    # identificar equipas
    equipas = df["equipa_jogador"].unique().to_list()
    if len(equipas) < 2:
        raise ValueError("É necessário haver pelo menos duas equipas distintas no dataset.")
    equipa_A, equipa_B = equipas[:2]

    # função auxiliar para calcular área convexa
    def area_convexa(xy: np.ndarray) -> float:
        if xy.shape[0] < 3:
            return 0.0
        try:
            return float(MultiPoint(xy).convex_hull.area)
        except Exception:
            return 0.0

    # iterar por frame
    for grupo in df.partition_by(["period", "timestamp"], maintain_order=True):
        pts_A = (
            grupo.filter(pl.col("equipa_jogador") == equipa_A)
            .select(["x_norm", "y_norm"])
            .drop_nulls()
            .to_numpy()
        )
        pts_B = (
            grupo.filter(pl.col("equipa_jogador") == equipa_B)
            .select(["x_norm", "y_norm"])
            .drop_nulls()
            .to_numpy()
        )

        area_A = area_convexa(pts_A)
        area_B = area_convexa(pts_B)

        # Evitar divisões por zero
        if area_B > 0:
            ratio = area_A / area_B
        else:
            ratio = 0.0

        resultados.append({
            "period": grupo["period"][0],
            "timestamp": grupo["timestamp"][0],
            "surface_area_ratio": ratio,
            "area_team_A": area_A,
            "area_team_B": area_B,
            "team_A": equipa_A,
            "team_B": equipa_B
        })

    return pl.DataFrame(resultados)
