# ==========================
# Team Centroid Distance (Distância entre centróides das equipas)
# ==========================
import polars as pl
import numpy as np

def calc_team_centroid_distance(df: pl.DataFrame) -> pl.DataFrame:
    resultados = []

    # garantir colunas obrigatórias
    if not {"equipa_jogador", "period", "timestamp", "x_norm", "y_norm"}.issubset(df.columns):
        raise ValueError("Faltam colunas obrigatórias: equipa_jogador, period, timestamp, x_norm, y_norm")

    # equipas únicas
    equipas = df["equipa_jogador"].unique().to_list()
    if len(equipas) < 2:
        raise ValueError("É necessário haver pelo menos duas equipas distintas no dataset.")

    equipa_A, equipa_B = equipas[:2]

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

        # calcular distância entre centróides
        if pts_A.shape[0] < 2 or pts_B.shape[0] < 2:
            dist = 0.0
        else:
            centro_A = pts_A.mean(axis=0)
            centro_B = pts_B.mean(axis=0)
            dist = float(np.linalg.norm(centro_A - centro_B))

        resultados.append({
            "period": grupo["period"][0],
            "timestamp": grupo["timestamp"][0],
            "team_centroid_distance": dist
        })

    return pl.DataFrame(resultados)
