# ==========================
# Team Spread (Dispersão)
# ==========================
import polars as pl
import numpy as np

def calc_team_spread(df: pl.DataFrame) -> pl.DataFrame:
    resultados = []

    # Verificar colunas necessárias
    if not {"x_norm", "y_norm", "team", "period", "timestamp"}.issubset(df.columns):
        raise ValueError("Faltam colunas obrigatórias: x_norm, y_norm, team, period, timestamp")

    # Iterar por equipa / período / timestamp
    for grupo in df.partition_by(["team", "period", "timestamp"], maintain_order=True):
        pts = grupo.select(["x_norm", "y_norm"]).drop_nulls().to_numpy()

        if pts.shape[0] < 2:
            spread = 0.0
        else:
            # centróide (média das coordenadas)
            centroid = pts.mean(axis=0)
            # distância euclidiana de cada jogador ao centróide
            dist = np.sqrt(((pts - centroid)**2).sum(axis=1))
            # dispersão média
            spread = float(np.mean(dist))

        resultados.append({
            "team": grupo["team"][0],
            "period": grupo["period"][0],
            "timestamp": grupo["timestamp"][0],
            "team_spread": spread
        })

    return pl.DataFrame(resultados)
