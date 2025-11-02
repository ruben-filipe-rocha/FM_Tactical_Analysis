# ==========================
# stretch_index.py (Polars)
# ==========================
import polars as pl

# ==========================
# 1) Adicionar Stretch Index a todos os frames (vectorizado)
# ==========================
def adicionar_stretch_index(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    keys = ["timestamp", "period", "equipa_jogador"]

    df_valid = df.filter(
        pl.col("x_norm").is_not_null() &
        pl.col("y_norm").is_not_null() &
        pl.col("equipa_jogador").is_not_null()
    )

    df_dist = (
        df_valid
        .with_columns([
            pl.col("x_norm").mean().over(keys).alias("mu_x"),
            pl.col("y_norm").mean().over(keys).alias("mu_y"),
        ])
        .with_columns([
            ((pl.col("x_norm") - pl.col("mu_x"))**2 + (pl.col("y_norm") - pl.col("mu_y"))**2)
            .sqrt()
            .alias("dist")
        ])
    )

    stretch_df = (
        df_dist.group_by(keys)
               .agg(pl.col("dist").mean().alias("stretch_index"))
    )

    out = df.join(stretch_df, on=keys, how="left")
    return out

# ==========================
# 2) Calcular Stretch Index de um frame/equipa (API compatível com o teu loop)
# ==========================
def calcular_stretch_index(df: pl.DataFrame | pl.LazyFrame,
                           timestamp: int | float,
                           period: int,
                           equipa: str) -> float | None:
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    sub = df.filter(
        (pl.col("timestamp") == timestamp) &
        (pl.col("period") == period) &
        (pl.col("equipa_jogador") == equipa) &
        pl.col("x_norm").is_not_null() &
        pl.col("y_norm").is_not_null()
    )

    if sub.height == 0:
        return None

    mu = sub.select([
        pl.col("x_norm").mean().alias("mu_x"),
        pl.col("y_norm").mean().alias("mu_y"),
    ])
    mu_x = mu["mu_x"][0]
    mu_y = mu["mu_y"][0]

    if mu_x is None or mu_y is None:
        return None

    val = sub.select(
        ((pl.col("x_norm") - pl.lit(mu_x))**2 + (pl.col("y_norm") - pl.lit(mu_y))**2)
        .sqrt()
        .mean()
        .alias("stretch")
    )["stretch"][0]

    return float(val) if val is not None else None
