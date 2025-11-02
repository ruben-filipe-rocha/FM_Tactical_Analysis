# ==========================
# detectar_lado.py (versão Polars)
# ==========================
import polars as pl

def obter_lado_equipa(df: pl.DataFrame | pl.LazyFrame) -> tuple[str, str]:
    # aceita DataFrame ou LazyFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # minuto 0 da 1.ª parte
    df_minuto_0 = df.filter((pl.col("period") == 1) & (pl.col("minute") == 0))
    if df_minuto_0.height == 0:
        raise ValueError("Sem registos no minuto 0 do período 1.")

    # timestamp inicial
    ts = df_minuto_0.select(pl.col("timestamp").min().alias("ts"))["ts"][0]

    # primeiro frame
    primeiro_frame = df_minuto_0.filter(pl.col("timestamp") == ts)
    if primeiro_frame.height == 0:
        raise ValueError("Sem frame inicial no timestamp mínimo.")

    # equipa do actor (quem executa o evento)
    equipa_actor = (
        primeiro_frame
        .filter(pl.col("actor") == True)
        .select(pl.col("team").first().alias("team"))
    )["team"][0]

    # equipa oponente
    equipas_unicas = df.select(pl.col("team").unique().alias("team"))["team"].to_list()
    if len(equipas_unicas) < 2:
        raise ValueError("Menos de duas equipas detetadas.")
    equipa_oponente = equipas_unicas[0] if equipas_unicas[0] != equipa_actor else equipas_unicas[1]

    # médias de x (teammate True vs False) no primeiro frame
    media_x_true = (
        primeiro_frame
        .filter((pl.col("teammate") == True) & pl.col("x").is_not_null())
        .select(pl.col("x").mean().alias("m"))
    )["m"][0]

    media_x_false = (
        primeiro_frame
        .filter((pl.col("teammate") == False) & pl.col("x").is_not_null())
        .select(pl.col("x").mean().alias("m"))
    )["m"][0]

    # determinar orientação
    if media_x_true < media_x_false:
        return equipa_actor, equipa_oponente  # esquerda, direita
    else:
        return equipa_oponente, equipa_actor
