# ==========================
# distancia_media.py (Polars) — fix com .implode()
# ==========================
import polars as pl
import numpy as np

def calcular_distancia_media(df_frame: pl.DataFrame) -> float | None:
    # média das distâncias par-a-par num frame/equipa
    sub = df_frame.select(["x_norm", "y_norm"]).drop_nulls()
    n = sub.height
    if n < 2:
        return None
    coords = sub.to_numpy()  # (n,2)
    diffs = coords[:, None, :] - coords[None, :, :]  # (n,n,2)
    dists = np.sqrt((diffs ** 2).sum(axis=2))        # (n,n)
    iu = np.triu_indices(n, 1)                       # i<j
    return float(dists[iu].mean())

def _mean_pairwise_from_lists(xs, ys):
    # xs/ys são listas de floats
    if xs is None or ys is None or len(xs) < 2:
        return None
    coords = np.column_stack([np.asarray(xs, float), np.asarray(ys, float)])
    n = coords.shape[0]
    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diffs ** 2).sum(axis=2))
    iu = np.triu_indices(n, 1)
    return float(dists[iu].mean())

def adicionar_distancia_media(df: pl.DataFrame | pl.LazyFrame,
                              equipa_esquerda=None, equipa_direita=None) -> pl.DataFrame:
    # agrega listas com .implode() (compatível com versões recentes do Polars)
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    keys = ["timestamp", "period", "equipa_jogador"]

    grouped = (
        lf.filter(
            pl.col("equipa_jogador").is_not_null() &
            pl.col("x_norm").is_not_null() &
            pl.col("y_norm").is_not_null()
        )
        .group_by(keys)
        .agg([
            pl.col("x_norm").implode().alias("xs"),  # lista de x por grupo
            pl.col("y_norm").implode().alias("ys"),  # lista de y por grupo
        ])
        .with_columns(
            pl.struct(["xs", "ys"])
              .map_elements(lambda s: _mean_pairwise_from_lists(s["xs"], s["ys"]), return_dtype=pl.Float64)
              .alias("distancia_media_entre_jogadores")
        )
        .select(keys + ["distancia_media_entre_jogadores"])
    )

    out = lf.join(grouped, on=keys, how="left").collect()
    return out
