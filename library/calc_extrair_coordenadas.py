
import polars as pl

def extrair_coordenadas_passes(df: pl.DataFrame | pl.LazyFrame):
    # x,y: 1º e 2º elemento de location_x ; pass_end_x,y: de pass_end_location
    # Assume colunas do tipo List(Float64). Se não forem List, ver célula 2.
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
