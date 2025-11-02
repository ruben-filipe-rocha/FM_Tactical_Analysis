# ==========================
# indice_equilibrio.py (Polars)
# ==========================
import polars as pl
from typing import Optional

# ==========================
# 1) Cálculo frame-a-frame (mesma lógica do Pandas)
# ==========================
def calcular_indice_equilibrio(df_frame: pl.DataFrame, equipa: str, direcao: str) -> Optional[float]:
    # x da bola (tem de existir 1 actor no frame)
    x_bola_list = (
        df_frame.filter(pl.col("actor") == True)
                .select("x_norm")
                .drop_nulls()
                .to_series()
                .to_list()
    )
    if len(x_bola_list) != 1:
        return None
    x_bola = float(x_bola_list[0])

    # jogadores da equipa com coordenadas válidas
    jogadores = (
        df_frame.filter(pl.col("equipa_jogador") == equipa)
                .select("x_norm")
                .drop_nulls()
    )
    total = jogadores.height
    if total == 0:
        return None

    # regra: equipa_esquerda conta atrás se x < x_bola; equipa_direita se x > x_bola
    if direcao == "direita":
        atras = jogadores.filter(pl.col("x_norm") < x_bola).height
    else:
        atras = jogadores.filter(pl.col("x_norm") > x_bola).height

    return (atras / total) * 100.0 if total > 0 else None

# ==========================
# 2) Aplicar a todas as equipas e frames (mesma API)
# ==========================
def adicionar_indice_equilibrio(df: pl.DataFrame | pl.LazyFrame,
                                equipa_esquerda: str,
                                equipa_direita: str) -> pl.DataFrame:
    df_pl = df.collect() if isinstance(df, pl.LazyFrame) else df

    # chaves dos frames
    frames = (
        df_pl.select(["timestamp", "period"])
             .unique()
             .sort(["period", "timestamp"])
             .iter_rows(named=True)
    )

    # acumular resultados
    equil_data = []
    for row in frames:
        ts, per = row["timestamp"], row["period"]
        frame = df_pl.filter((pl.col("timestamp") == ts) & (pl.col("period") == per))

        # equipa à esquerda avalia "direita" (atrás: x < x_bola)
        eq_esq = calcular_indice_equilibrio(frame, equipa_esquerda, "direita")
        equil_data.append((ts, per, equipa_esquerda, eq_esq))

        # equipa à direita avalia "esquerda" (atrás: x > x_bola)
        eq_dir = calcular_indice_equilibrio(frame, equipa_direita, "esquerda")
        equil_data.append((ts, per, equipa_direita, eq_dir))

    # resultados e join
    equil_df = pl.DataFrame(
        equil_data,
        schema=["timestamp", "period", "equipa_jogador", "indice_equilibrio_defensivo"]
    )

    out = df_pl.join(equil_df, on=["timestamp", "period", "equipa_jogador"], how="left")
    return out
