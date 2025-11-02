# ==========================
# compactacao.py (Polars)
# ==========================
import polars as pl

# ==========================
# 1) Calcular compactação de um frame (opcional, API compatível)
# ==========================
def calcular_compactacao(df_frame: pl.DataFrame) -> float | None:
    # usa x_norm e y_norm; devolve área do bounding box ou None
    sub = df_frame.filter(pl.col("x_norm").is_not_null() & pl.col("y_norm").is_not_null())
    if sub.height == 0:
        return None
    stats = sub.select(
        pl.col("x_norm").min().alias("x_min"),
        pl.col("x_norm").max().alias("x_max"),
        pl.col("y_norm").min().alias("y_min"),
        pl.col("y_norm").max().alias("y_max"),
    )
    x_min, x_max, y_min, y_max = stats.row(0)
    if None in (x_min, x_max, y_min, y_max):
        return None
    return float((x_max - x_min) * (y_max - y_min))

# ==========================
# 2) Adicionar compactação a todos os frames (vectorizado)
# ==========================
def adicionar_compactacao(df: pl.DataFrame | pl.LazyFrame,
                          equipa_esquerda=None, equipa_direita=None) -> pl.DataFrame:
    # equipa_esquerda/direita mantidos para compatibilidade; não usados
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    keys = ["timestamp", "period", "equipa_jogador"]

    compact_df = (
        lf.filter(pl.col("x_norm").is_not_null() & pl.col("y_norm").is_not_null() & pl.col("equipa_jogador").is_not_null())
          .group_by(keys)
          .agg([
              pl.col("x_norm").min().alias("x_min"),
              pl.col("x_norm").max().alias("x_max"),
              pl.col("y_norm").min().alias("y_min"),
              pl.col("y_norm").max().alias("y_max"),
          ])
          .with_columns([
              (pl.col("x_max") - pl.col("x_min")).alias("comprimento"),
              (pl.col("y_max") - pl.col("y_min")).alias("largura"),
              ((pl.col("x_max") - pl.col("x_min")) * (pl.col("y_max") - pl.col("y_min"))).alias("compactacao"),
          ])
          .select(keys + ["compactacao"])
    )

    out = (
        lf.join(compact_df, on=keys, how="left")
          .collect()
    )
    return out
