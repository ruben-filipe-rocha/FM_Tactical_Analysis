# ==========================
# quebra_linha.py (Polars)
# ==========================
import polars as pl
import numpy as np
from typing import Optional, Mapping, Any, List, Tuple

# ==========================
# 1. Função base (mesma lógica do Pandas)
# ==========================
def quebra_linha(
    passe_row: Mapping[str, Any],
    df_frame: pl.DataFrame,
    equipa_esquerda: str,
    equipa_direita: str,
    direcao: str
) -> bool:
    # equipa atacante e defensora
    equipa_atacante = passe_row["equipa_jogador"]
    equipa_defensora = equipa_direita if equipa_atacante == equipa_esquerda else equipa_esquerda

    # destino do passe
    x_destino = passe_row["pass_end_x"]
    if x_destino is None or (isinstance(x_destino, float) and np.isnan(x_destino)):
        return False

    # defensores no frame
    defensores_x = (
        df_frame
        .filter(pl.col("equipa_jogador") == equipa_defensora)
        .select("x_norm")
        .drop_nulls()
        .to_series()
        .to_list()
    )
    if len(defensores_x) == 0:
        return False

    if direcao == "direita":
        linha_defensiva = max(defensores_x)
        return x_destino > linha_defensiva
    else:
        linha_defensiva = min(defensores_x)
        return x_destino < linha_defensiva

# ==========================
# 2. Aplicar a todos os passes (mesma API, agora em Polars)
# ==========================
def adicionar_passes_quebram_linha(
    df: pl.DataFrame | pl.LazyFrame,
    equipa_esquerda: str,
    equipa_direita: str
) -> pl.DataFrame:
    df_pl = df.collect() if isinstance(df, pl.LazyFrame) else df

    # listar passes
    passes: List[Mapping[str, Any]] = (
        df_pl
        .filter(pl.col("type") == "Pass")
        .select(["timestamp", "period", "equipa_jogador", "pass_end_x"])
        .to_dicts()
    )

    # calcular por passe
    resultados: List[Tuple[Any, Any, str, int]] = []
    for p in passes:
        ts, per = p["timestamp"], p["period"]
        df_frame = df_pl.filter((pl.col("timestamp") == ts) & (pl.col("period") == per))
        direcao = "direita" if p["equipa_jogador"] == equipa_esquerda else "esquerda"
        quebrou = quebra_linha(p, df_frame, equipa_esquerda, equipa_direita, direcao)
        resultados.append((ts, per, p["equipa_jogador"], int(bool(quebrou))))

    # resultados e merge
    passes_quebra_df = pl.DataFrame(
        resultados,
        schema=["timestamp", "period", "equipa_jogador", "passe_quebra_linha"]
    )

    out = (
        df_pl
        .drop("passe_quebra_linha", strict=False)  # evitar duplicação
        .join(passes_quebra_df, on=["timestamp", "period", "equipa_jogador"], how="left")
    )
    return out
