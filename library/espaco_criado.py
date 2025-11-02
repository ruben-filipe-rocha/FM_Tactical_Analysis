# ==========================
# espaco_criado.py (Polars)
# ==========================
import polars as pl

def adicionar_espaco_criado(df: pl.DataFrame | pl.LazyFrame, area_total: float = 120.0 * 80.0) -> pl.DataFrame:
    # requer 'compactacao'
    if "compactacao" not in df.columns:
        raise ValueError("A coluna 'compactacao' é necessária. Calcula-a antes de aplicar esta função.")
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df
    # espaco_criado = área total - compactacao ; keeper = False
    return (
        lf.with_columns([
            (pl.lit(area_total) - pl.col("compactacao")).alias("espaco_criado"),
            pl.lit(False).alias("keeper"),
        ])
        .collect()
    )
