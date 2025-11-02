# ==========================
# 1. Imports
# ==========================
import ast
import numpy as np
import plotly.graph_objects as go
import polars as pl
from grid_hex import gerar_hex_grid  # importa a grelha

# ==========================
# 2. Detectar equipa à esquerda (Polars; aceita Pandas/Polars)
# ==========================
def obter_lado_equipa(df_any) -> tuple[str, str]:
    if isinstance(df_any, pl.LazyFrame):
        df = df_any.collect()
    elif isinstance(df_any, pl.DataFrame):
        df = df_any
    else:
        df = pl.from_pandas(df_any)

    team_col = "team" if "team" in df.columns else "equipa_jogador"
    x_col = "x" if "x" in df.columns else ("x_norm" if "x_norm" in df.columns else None)
    if x_col is None:
        raise ValueError("Falta 'x' ou 'x_norm'.")

    df_p1 = df.filter(pl.col("period") == 1)
    if df_p1.height == 0:
        raise ValueError("Sem registos no período 1.")

    base = df_p1
    if "minute" in df_p1.columns and df_p1.filter(pl.col("minute") == 0).height > 0:
        base = df_p1.filter(pl.col("minute") == 0)

    ts = base.select(pl.col("timestamp").min().alias("ts"))["ts"][0]
    primeiro_frame = df_p1.filter(pl.col("timestamp") == ts)
    if primeiro_frame.height == 0:
        raise ValueError("Sem frame inicial.")

    equipa_actor = (
        primeiro_frame.filter(pl.col("actor") == True)
        .select(pl.col(team_col).first().alias("team"))
    )["team"][0]

    equipas_unicas = df.select(pl.col(team_col).unique()).to_series().to_list()
    if len(equipas_unicas) < 2:
        raise ValueError("Menos de duas equipas.")
    equipa_oponente = equipas_unicas[0] if equipas_unicas[0] != equipa_actor else equipas_unicas[1]

    media_x_true = (
        primeiro_frame.filter((pl.col("teammate") == True) & pl.col(x_col).is_not_null())
        .select(pl.col(x_col).mean().alias("m"))
    )["m"][0]
    media_x_false = (
        primeiro_frame.filter((pl.col("teammate") == False) & pl.col(x_col).is_not_null())
        .select(pl.col(x_col).mean().alias("m"))
    )["m"][0]

    if media_x_true < media_x_false:
        return equipa_actor, equipa_oponente
    else:
        return equipa_oponente, equipa_actor

# ==========================
# 3. Polígono da área visível
# ==========================
def _visible_polygon_from_row(row, do_flip=False):
    arr = row.get("visible_area")
    if arr is None:
        return go.Scatter(x=[], y=[], mode="lines",
                          line=dict(color="red", width=2),
                          fill="toself", fillcolor="rgba(255,0,0,0.25)",
                          hoverinfo="skip", showlegend=False)
    if isinstance(arr, str):
        try:
            arr = ast.literal_eval(arr)
        except Exception:
            arr = None
    if not isinstance(arr, (list, tuple, np.ndarray)):
        return go.Scatter(x=[], y=[], mode="lines",
                          line=dict(color="red", width=2),
                          fill="toself", fillcolor="rgba(255,0,0,0.25)",
                          hoverinfo="skip", showlegend=False)

    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size < 6 or (arr.size % 2 != 0):
        return go.Scatter(x=[], y=[], mode="lines",
                          line=dict(color="red", width=2),
                          fill="toself", fillcolor="rgba(255,0,0,0.25)",
                          hoverinfo="skip", showlegend=False)

    x = arr[::2].tolist()
    y = arr[1::2].tolist()

    if do_flip:
        x = [120 - xi for xi in x]
        y = [80  - yi for yi in y]

    if x[0] != x[-1] or y[0] != y[-1]:
        x.append(x[0]); y.append(y[0])

    return go.Scatter(
        x=x, y=y, mode="lines",
        fill="toself", fillcolor="rgba(255,0,0,0.25)",
        line=dict(color="red", width=2),
        hoverinfo="skip", showlegend=False
    )

# ==========================
# 4. Campo (estático)
# ==========================
def _desenhar_campo(fig):
    fig.add_shape(type="rect", x0=-3, y0=-6, x1=123, y1=86,
                  fillcolor="green", line=dict(width=0), layer="below")
    for i in range(6):
        x0 = i * (120/6); x1 = x0 + (120/6)
        fig.add_shape(type="rect", x0=x0, y0=0, x1=x1, y1=80,
                      fillcolor=f"rgba(255,255,255,{0.03 if i%2==0 else 0.0})",
                      line=dict(width=0), layer="below")
    def linha(x1, y1, x2, y2, w=2):
        fig.add_trace(go.Scatter(x=[x1, x2], y=[y1, y2], mode="lines",
                                 line=dict(color="white", width=w),
                                 hoverinfo="skip", showlegend=False))
    linha(0, 0, 120, 0); linha(120, 0, 120, 80); linha(120, 80, 0, 80); linha(0, 80, 0, 0)
    linha(60, 0, 60, 80)
    for a in np.linspace(0, 2*np.pi, 60, endpoint=False):
        fig.add_trace(go.Scatter(
            x=[60 + 9.15*np.cos(a), 60 + 9.15*np.cos(a + 2*np.pi/60)],
            y=[40 + 9.15*np.sin(a), 40 + 9.15*np.sin(a + 2*np.pi/60)],
            mode="lines", line=dict(color="white", width=2),
            hoverinfo="skip", showlegend=False))
    linha(0, 18, 18, 18); linha(18, 18, 18, 62); linha(18, 62, 0, 62)
    linha(102, 18, 120, 18); linha(102, 18, 102, 62); linha(120, 62, 102, 62)
    linha(0, 30, 6, 30); linha(6, 30, 6, 50); linha(6, 50, 0, 50)
    linha(114, 30, 120, 30); linha(114, 30, 114, 50); linha(114, 50, 120, 50)
    for xp, yp in [(11, 40), (109, 40)]:
        fig.add_trace(go.Scatter(x=[xp], y=[yp], mode="markers",
                                 marker=dict(color="white", size=4),
                                 hoverinfo="skip", showlegend=False))
    raio = 9.15
    ang_esq = np.linspace(-np.pi/3, np.pi/3, 80)
    x_esq = [11 + raio*np.cos(a) for a in ang_esq if 11 + raio*np.cos(a) > 18]
    y_esq = [40 + raio*np.sin(a) for a in ang_esq if 11 + raio*np.cos(a) > 18]
    fig.add_trace(go.Scatter(x=x_esq, y=y_esq, mode="lines",
                             line=dict(color="white", width=2),
                             hoverinfo="skip", showlegend=False))
    ang_dir = np.linspace(2*np.pi/3, 4*np.pi/3, 80)
    x_dir = [109 + raio*np.cos(a) for a in ang_dir if 109 + raio*np.cos(a) < 102]
    y_dir = [40 + raio*np.sin(a) for a in ang_dir if 109 + raio*np.cos(a) < 102]
    fig.add_trace(go.Scatter(x=x_dir, y=y_dir, mode="lines",
                             line=dict(color="white", width=2),
                             hoverinfo="skip", showlegend=False))
    for xg, s in [(0, -1), (120, 1)]:
        linha(xg, 36, xg + s*2, 36); linha(xg, 44, xg + s*2, 44); linha(xg + s*2, 36, xg + s*2, 44)
    fig.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80,
                  line=dict(color="white", width=3), fillcolor="rgba(0,0,0,0)", layer="above")

# ==========================
# 5. Animação (com grelha sobreposta)
# ==========================
def criar_animacao_plotly_full(df_minuto, ranked_timestamps, equipa_azul, equipa_vermelha, equipa_esquerda_nome=None):
    if equipa_esquerda_nome is None:
        esquerda, _ = obter_lado_equipa(df_minuto)
        equipa_esquerda_nome = esquerda

    fig = go.Figure()

    # Traces animados
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                             line=dict(color="red", width=2),
                             fill="toself", fillcolor="rgba(255,0,0,0.25)",
                             name="Área Visível", hoverinfo="skip", showlegend=False))  # 0
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                             marker=dict(color="blue", size=9, line=dict(color="white", width=1)),
                             name=f"{equipa_azul}"))  # 1
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                             marker=dict(color="red", size=9, line=dict(color="white", width=1)),
                             name=f"{equipa_vermelha}"))  # 2
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                             marker=dict(color="white", size=10, line=dict(color="black", width=1)),
                             name="Bola"))  # 3

    # Campo
    _desenhar_campo(fig)

    # --- Grelha hexagonal sobreposta ---
    hexes = gerar_hex_grid(field_width=120, field_height=80, hex_size=1.84)
    for hex_coords in hexes:
        x, y = zip(*hex_coords)
        fig.add_trace(go.Scatter(
            x=x + (x[0],),
            y=y + (y[0],),
            mode="lines",
            line=dict(color="rgba(255,255,255,0.25)", width=0.8),
            hoverinfo="skip",
            showlegend=False
        ))

    # Frames e slider (igual ao original)
    frames, slider_steps = [], []

    def fmt_ts(v):
        import pandas as pd
        if isinstance(v, pd.Timestamp):
            total = v.minute*60 + v.second
        else:
            total = int(pd.to_timedelta(v).total_seconds())
        m, s = divmod(total, 60)
        return f"{m:02d}:{s:02d}"

    for _, r in ranked_timestamps.iterrows():
        ts, period = r["timestamp"], r["period"]
        fdf = df_minuto[(df_minuto["timestamp"] == ts) & (df_minuto["period"] == period)]
        if fdf.empty:
            continue

        df_A = fdf[fdf["teammate"] == True]
        df_B = fdf[fdf["teammate"] == False]
        df_actor = fdf[fdf["actor"] == True]

        if "equipa_cbola_nome" in fdf.columns:
            pos_nome = fdf.iloc[0]["equipa_cbola_nome"]
        elif "equipa_cbola" in fdf.columns:
            pos_nome = fdf.iloc[0]["equipa_cbola"]
        elif not df_actor.empty and "team" in df_actor.columns:
            pos_nome = df_actor.iloc[0]["team"]
        elif not df_actor.empty and "equipa_jogador" in df_actor.columns:
            pos_nome = df_actor.iloc[0]["equipa_jogador"]
        else:
            pos_nome = None
        do_flip = (pos_nome is not None and pos_nome != equipa_esquerda_nome)

        xb = yb = nome = None
        if not df_actor.empty:
            xb = min(float(df_actor.iloc[0]["x_norm"]) + 1.5, 120.0)
            yb = float(df_actor.iloc[0]["y_norm"])
            nome = df_actor.iloc[0].get("player", "")

        poly = _visible_polygon_from_row(df_actor.iloc[0], do_flip) if not df_actor.empty else \
               go.Scatter(x=[], y=[], mode="lines",
                          line=dict(color="red", width=2),
                          fill="toself", fillcolor="rgba(255,0,0,0.25)",
                          hoverinfo="skip", showlegend=False)

        frame_data = [
            poly,
            go.Scatter(x=df_A["x_norm"], y=df_A["y_norm"], mode="markers",
                       marker=dict(color="blue", size=9, line=dict(color="white", width=1))),
            go.Scatter(x=df_B["x_norm"], y=df_B["y_norm"], mode="markers",
                       marker=dict(color="red", size=9, line=dict(color="white", width=1))),
            go.Scatter(x=[xb] if xb is not None else [], y=[yb] if yb is not None else [],
                       mode="markers", marker=dict(color="white", size=10, line=dict(color="black", width=1)))
        ]

        annotations = []
        if nome and xb is not None and yb is not None:
            annotations.append(dict(x=xb, y=yb+2, text=nome, showarrow=False,
                                    font=dict(color="white", size=11),
                                    bgcolor="rgba(0,0,0,0.5)",
                                    bordercolor="white", borderwidth=1))

        frames.append(go.Frame(
            name=fmt_ts(ts),
            data=frame_data,
            layout=go.Layout(title=f"Período {period} | Timestamp {fmt_ts(ts)}", annotations=annotations)
        ))

        slider_steps.append({
            "args": [[fmt_ts(ts)], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate", "transition": {"duration": 0}}],
            "label": fmt_ts(ts), "method": "animate"
        })

    sliders = [{
        "active": 0, "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"color": "black"}, "prefix": "", "visible": True},
        "pad": {"b": 10, "t": 35}, "len": 0.9, "x": 0.05, "y": -0.06,
        "steps": slider_steps
    }]

    fig.update_layout(
        xaxis=dict(range=[-1, 121], showgrid=False, zeroline=False, ticks="", showticklabels=False, showline=False),
        yaxis=dict(range=[-3, 83], showgrid=False, zeroline=False, ticks="", showticklabels=False, showline=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="#d3d3d3", paper_bgcolor="#d3d3d3",
        width=1050, height=620, title="Evolução do jogo - análise tática",
        sliders=sliders,
        updatemenus=[{"type": "buttons", "x": 1.06, "y": 0.06,
                      "buttons": [
                          {"label": "▶ Play", "method": "animate",
                           "args": [None, {"frame": {"duration": 350, "redraw": True}, "fromcurrent": True}]},
                          {"label": "⏸ Pause", "method": "animate",
                           "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
                      ]}]
    )

    if frames:
        fig.frames = frames
    fig.show()
