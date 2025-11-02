# ==========================
# 1. Imports
# ==========================
import polars as pl

# ==========================
# 2. Função principal
# ==========================
def calc_pressing(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calcula a métrica de Pressing por equipa.
    Critério:
      - Contar eventos sob pressão (`under_pressure == True`)
        e/ou de contra-pressão (`counterpress == True`)
      - Agrupar por equipa e período
      - Normalizar por minuto de jogo (para intensidade comparável)
    """

    # garantir colunas necessárias
    colunas = {"team", "period", "minute", "under_pressure", "counterpress"}
    faltam = colunas - set(df.columns)
    if faltam:
        raise ValueError(f"Faltam colunas no DF: {faltam}")

    # Converter booleanos True/False em 0/1
    df_prep = df.select([
        pl.col("team"),
        pl.col("period"),
        pl.col("minute"),
        (pl.col("under_pressure").cast(pl.Int64) +
         pl.col("counterpress").cast(pl.Int64)).alias("press_events")
    ])

    # Agrupar por equipa e período
    df_press = (
        df_prep.group_by(["team", "period"])
        .agg([
            pl.sum("press_events").alias("press_count"),
            pl.count().alias("total_events"),
            pl.col("minute").max().alias("max_minute")
        ])
        .with_columns([
            # intensidade = nº pressões / minutos jogados
            (pl.col("press_count") / (pl.col("max_minute") + 1)).alias("press_intensity")
        ])
        .select(["team", "period", "press_count", "press_intensity"])
    )

    return df_press
