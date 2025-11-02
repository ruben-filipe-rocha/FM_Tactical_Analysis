# ==========================
# distancia_coletiva.py (Polars)
# ==========================
import polars as pl
import numpy as np
from typing import Optional

# ==========================
# 1. Calcular distância média entre duas equipas (frame)
# ==========================
def calcular_distancia_entre_equipas(df_a: pl.DataFrame, df_b: pl.DataFrame) -> Optional[float]:
    # limpar coordenadas inválidas
    a = df_a.select(["x_norm", "y_norm"]).drop_nulls()
    b = df_b.select(["x_norm", "y_norm"]).drop_nulls()
    if a.height == 0 or b.height == 0:
        return None

    # arrays (n_a,2) e (n_b,2)
    A = a.to_numpy()
    B = b.to_numpy()

    # distâncias todas vs todas com broadcasting
    diffs = A[:, None, :] - B[None, :, :]         # (n_a, n_b, 2)
    dists = np.sqrt((diffs ** 2).sum(axis=2))     # (n_a, n_b)
    return float(dists.mean())

# ==========================
# 2. Aplicar a todos os frames e equipas
# ==========================
def adicionar_distancia_coletiva(df: pl.DataFrame | pl.LazyFrame,
                                 equipa_esquerda: str,
                                 equipa_direita: str) -> pl.DataFrame:
    df_pl = df.collect() if isinstance(df, pl.LazyFrame) else df

    # iterar frames (timestamp, period)
    frames = (
        df_pl.select(["timestamp", "period"])
             .unique()
             .sort(["period", "timestamp"])
             .iter_rows(named=True)
    )

    registos = []
    for row in frames:
        ts, per = row["timestamp"], row["period"]
        frame = df_pl.filter((pl.col("timestamp") == ts) & (pl.col("period") == per))

        df_esq = frame.filter(pl.col("equipa_jogador") == equipa_esquerda)
        df_dir = frame.filter(pl.col("equipa_jogador") == equipa_direita)

        dist_media = calcular_distancia_entre_equipas(df_esq, df_dir)

        # associar o mesmo valor às duas equipas no frame
        registos.append((ts, per, equipa_esquerda, dist_media))
        registos.append((ts, per, equipa_direita, dist_media))

    dist_df = pl.DataFrame(
        registos,
        schema=["timestamp", "period", "equipa_jogador", "distancia_coletiva_entre_equipas"]
    )

    # merge final
    out = df_pl.join(dist_df, on=["timestamp", "period", "equipa_jogador"], how="left")
    return out
