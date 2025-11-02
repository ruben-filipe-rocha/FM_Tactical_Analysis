# ==========================
# 2) Função robusta (Polars) — não inverte quando equipa_actor_frame é null
# ==========================
import polars as pl

def normalizar_coordenadas(df: pl.DataFrame, equipa_esquerda: str, field_length: float = 120.0, field_width: float = 80.0) -> pl.DataFrame:
    # garantir numérico
    df = df.with_columns([
        pl.col("x").cast(pl.Float64, strict=False),
        pl.col("y").cast(pl.Float64, strict=False)
    ])

    # criar máscara lógica
    mask = (pl.col("equipa_jogador").is_not_null()) & (pl.col("equipa_jogador") != equipa_esquerda)

    # aplicar normalização
    df = df.with_columns([
        pl.when(mask).then(field_length - pl.col("x")).otherwise(pl.col("x")).alias("x_norm"),
        pl.when(mask).then(field_width  - pl.col("y")).otherwise(pl.col("y")).alias("y_norm")
    ])

    return df
