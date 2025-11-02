# preprocess_utils.py

import ast

def converter_visible_area(df):
    visible_area = df['visible_area'].dropna().apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
    return visible_area.apply(lambda lst: list(zip(lst[::2], lst[1::2])))


#

def extrair_coordenadas_passes(df):
    df["x"] = df["location_x"].apply(lambda loc: loc[0] if isinstance(loc, list) and len(loc) > 0 else None)
    df["y"] = df["location_x"].apply(lambda loc: loc[1] if isinstance(loc, list) and len(loc) > 1 else None)
    df["pass_end_x"] = df["pass_end_location"].apply(lambda loc: loc[0] if isinstance(loc, list) and len(loc) > 0 else None)
    df["pass_end_y"] = df["pass_end_location"].apply(lambda loc: loc[1] if isinstance(loc, list) and len(loc) > 1 else None)
    return df

