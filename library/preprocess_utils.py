# ==========================
# preprocess_utils.py (Polars-only)
# ==========================
import ast
import polars as pl

# ==========================
# 1) Converter 'visible_area' → pares [(x1,y1), (x2,y2), ...]
# ==========================
def converter_visible_area(df: pl.DataFrame) -> pl.Series:
    # aceita strings tipo "[x1, y1, x2, y2, ...]" ou listas; devolve Series de listas de pares
    s_list = df["visible_area"].map_elements(
        lambda v: ast.literal_eval(v) if isinstance(v, str) else v,
        return_dtype=pl.List(pl.Float64),
    )
    s_pairs = s_list.map_elements(
        lambda lst: None if lst is None else [[lst[i], lst[i+1]] for i in range(0, len(lst) - 1, 2)],
        return_dtype=pl.List(pl.List(pl.Float64)),
    )
    return s_pairs

# ==========================
# 2) Extrair coordenadas de passes (location_x → x,y ; pass_end_location → pass_end_x,y)
# ==========================
def extrair_coordenadas_passes(df: pl.DataFrame | pl.LazyFrame):
    # requer colunas List(Float64); funciona em DataFrame e LazyFrame
    return df.with_columns([
        pl.when(pl.col("location_x").is_not_null() & (pl.col("location_x").list.len() >= 1))
          .then(pl.col("location_x").list.get(0).cast(pl.Float64))
          .otherwise(None).alias("x"),
        pl.when(pl.col("location_x").is_not_null() & (pl.col("location_x").list.len() >= 2))
          .then(pl.col("location_x").list.get(1).cast(pl.Float64))
          .otherwise(None).alias("y"),
        pl.when(pl.col("pass_end_location").is_not_null() & (pl.col("pass_end_location").list.len() >= 1))
          .then(pl.col("pass_end_location").list.get(0).cast(pl.Float64))
          .otherwise(None).alias("pass_end_x"),
        pl.when(pl.col("pass_end_location").is_not_null() & (pl.col("pass_end_location").list.len() >= 2))
          .then(pl.col("pass_end_location").list.get(1).cast(pl.Float64))
          .otherwise(None).alias("pass_end_y"),
    ])
