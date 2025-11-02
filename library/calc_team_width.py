# ==========================
# Team Width (Largura da equipa)
# ==========================
import polars as pl
import numpy as np

def calc_team_width(df: pl.DataFrame) -> pl.DataFrame:
    resultados = []

    # Verificar colunas obrigatórias
    if not {"y_norm", "team", "period", "timestamp"}.issubset(df.columns):
        raise ValueError("Faltam colunas obrigatórias: y_norm, team, period, timestamp")

    # Iterar por equipa / período / timestamp
    for grupo in df.partition_by(["team", "period", "timestamp"], maintain_order=True):
        ys = grupo["y_norm"].drop_nulls().to_numpy()

        if ys.size < 2:
            width = 0.0
        else:
            width = float(ys.max() - ys.min())

        resultados.append({
            "team": grupo["team"][0],
            "period": grupo["period"][0],
            "timestamp": grupo["timestamp"][0],
            "team_width": width
        })

    return pl.DataFrame(resultados)
