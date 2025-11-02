# ==========================
# corrigir_team.py (Polars) — fix sem colunas intermed.
# ==========================
import polars as pl

def corrigir_team_por_frame(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    # aceita LazyFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # equipas do jogo (assume 2)
    equipas = (
        df.select(pl.col("team").drop_nulls().unique())
          .to_series(0)
          .to_list()
    )
    if len(equipas) != 2:
        raise ValueError("Esperavam-se exatamente 2 equipas.")
    t1, t2 = equipas[0], equipas[1]

    # equipa do actor por (period, timestamp)
    actor_por_frame = (
        df.group_by(["period", "timestamp"])
          .agg(pl.col("team").filter(pl.col("actor") == True).first().alias("equipa_actor_frame"))
    )

    # join + team_corrigido (sem referenciar colunas criadas no mesmo with_columns)
    out = (
        df.join(actor_por_frame, on=["period", "timestamp"], how="left")
          .with_columns([
              pl.when(pl.col("equipa_actor_frame").is_not_null())
                .then(
                    pl.when(pl.col("teammate") == True)
                      .then(pl.col("equipa_actor_frame"))
                      .otherwise(
                          pl.when(pl.col("equipa_actor_frame") == pl.lit(t1)).then(pl.lit(t2)).otherwise(pl.lit(t1))
                      )
                )
                .otherwise(None)
                .alias("equipa_jogador")
          ])
    )
    return out
