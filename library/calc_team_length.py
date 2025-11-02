# ==========================
# Team Length (Extensão longitudinal)
# ==========================
import polars as pl
import numpy as np

def calc_team_length(df: pl.DataFrame) -> pl.DataFrame:
    resultados = []

    # Verificar colunas obrigatórias
    if not {"x_norm", "team", "period", "timestamp"}.issubset(df.columns):
        raise ValueError("Faltam colunas obrigatórias: x_norm, team, period, timestamp")

    # Iterar por equipa / período / timestamp
    for grupo in df.partition_by(["team", "period", "timestamp"], maintain_order=True):
        xs = grupo["x_norm"].drop_nulls().to_numpy()

        if xs.size < 2:
            length = 0.0
        else:
            length = float(xs.max() - xs.min())

        resultados.append({
            "team": grupo["team"][0],
            "period": grupo["period"][0],
            "timestamp": grupo["timestamp"][0],
            "team_length": length
        })

    return pl.DataFrame(resultados)
