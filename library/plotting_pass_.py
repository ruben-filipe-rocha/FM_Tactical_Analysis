# ==========================
# 1. Imports
# ==========================
import ast
import numpy as np
import plotly.graph_objects as go
import polars as pl
from quebra_linha import quebra_linha  # adicionado

# ==========================
# 2. Detectar equipa à esquerda (Polars; aceita Pandas/Polars)
# ==========================
def obter_lado_equipa(df_any) -> tuple[str, str]:
    # aceita pd.DataFrame, pl.DataFrame ou pl.LazyFrame
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
# 3. Polígono da área visível (flip 180º se do_flip=True)
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
# 5. Animação (com passe e deteção de quebra de linha)
# ==========================
def criar_animacao_plotly_full(df_minuto, ranked_timestamps, equipa_azul, equipa_vermelha, equipa_esquerda_nome=None):
    if equipa_esquerda_nome is None:
        esquerda, _ = obter_lado_equipa(df_minuto)
        equipa_esquerda_nome = esquerda

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=[], y=[], mode="lines",
                         line=dict(color="red", width=2),
                         fill="toself", fillcolor="rgba(255,0,0,0.25)",
                         hoverinfo="skip", showlegend=False))  # 0
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                            marker=dict(color="blue", size=9, line=dict(color="white", width=1)),
                            hoverinfo="skip", showlegend=False))  # 1
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                            marker=dict(color="red", size=9, line=dict(color="white", width=1)),
                            hoverinfo="skip", showlegend=False))  # 2
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers",
                            marker=dict(color="white", size=10, line=dict(color="black", width=1)),
                            hoverinfo="skip", showlegend=False))  # 3


    _desenhar_campo(fig)
    frames, slider_steps = [], []

    def fmt_ts(v):
        import pandas as pd
        if isinstance(v, pd.Timestamp):
            total = v.minute * 60 + v.second
        else:
            total = int(pd.to_timedelta(v).total_seconds())
        m, s = divmod(total, 60)
        return f"{m:02d}:{s:02d}"

    for _, r in ranked_timestamps.iterrows():
        ts, period = r["timestamp"], r["period"]
        frame_data = df_minuto[(df_minuto["timestamp"] == ts) & (df_minuto["period"] == period)]
        if frame_data.empty:
            continue

        df_A = frame_data[frame_data["teammate"] == True]
        df_B = frame_data[frame_data["teammate"] == False]
        df_actor = frame_data[frame_data["actor"] == True]

        # posse no frame → decide flip
        if "equipa_cbola_nome" in frame_data.columns:
            pos_nome = frame_data.iloc[0]["equipa_cbola_nome"]
        elif "equipa_cbola" in frame_data.columns:
            pos_nome = frame_data.iloc[0]["equipa_cbola"]
        elif not df_actor.empty and "team" in df_actor.columns:
            pos_nome = df_actor.iloc[0]["team"]
        elif not df_actor.empty and "equipa_jogador" in df_actor.columns:
            pos_nome = df_actor.iloc[0]["equipa_jogador"]
        else:
            pos_nome = None
        do_flip = (pos_nome is not None and pos_nome != equipa_esquerda_nome)

        # bola e nome (opcional)
        xb = yb = nome = None
        if not df_actor.empty:
            xb = min(float(df_actor.iloc[0]["x_norm"]) + 1.5, 120.0)
            yb = float(df_actor.iloc[0]["y_norm"])
            nome = df_actor.iloc[0].get("player", "")

        # 0: polígono da área visível
        poly = _visible_polygon_from_row(df_actor.iloc[0], do_flip) if not df_actor.empty else \
               go.Scatter(x=[], y=[], mode="lines",
                          line=dict(color="red", width=2),
                          fill="toself", fillcolor="rgba(255,0,0,0.25)",
                          hoverinfo="skip", showlegend=False)

       # ==========================
        # 5.1 Passe do actor com deteção de quebra de linha
        # ==========================
        import pandas as pd

        linha_passe = go.Scatter(x=[], y=[], mode="lines",
                                line=dict(color="white", width=1.5),
                                hoverinfo="skip", showlegend=False)

        # Selecionar o passe executado pelo jogador em posse
        passe_actor = frame_data[(frame_data["type"] == "Pass") & (frame_data["actor"] == True)]
        if not passe_actor.empty:
            passe = passe_actor.iloc[0]

            # Coordenadas do passe (origem e destino)
            x0, y0 = float(passe["x"]), float(passe["y"])
            x1, y1 = float(passe["pass_end_x"]), float(passe["pass_end_y"])

            # Aplicar flip quando equipa com bola ataca da direita
            if do_flip:
                x0, x1 = 120 - x0, 120 - x1
                y0, y1 = 80 - y0, 80 - y1

            if not any(pd.isna([x0, y0, x1, y1])):
                equipa_direita = equipa_vermelha if equipa_esquerda_nome == equipa_azul else equipa_azul
                direcao = "direita" if passe["equipa_jogador"] == equipa_esquerda_nome else "esquerda"

                # Verificar quebra de linha
                is_quebra = quebra_linha(
                    passe_row=passe,
                    df_frame=pl.from_pandas(frame_data),
                    equipa_esquerda=equipa_esquerda_nome,
                    equipa_direita=equipa_direita,
                    direcao=direcao
                )

                cor = "red" if is_quebra else "white"

                # Linha + seta
                linha_passe = go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color=cor, width=1.4, dash="solid", shape="spline", smoothing=1.3),
                    opacity=0.9,
                    hoverinfo="skip",
                    showlegend=False
                )

                arrow_annotation = dict(
                    ax=x0, ay=y0, x=x1, y=y1,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3,       # seta pontiaguda e discreta
                    arrowsize=0.9,     # tamanho proporcional (antes estava 2.4 → muito grande)
                    arrowwidth=1.6,    # mais fino e elegante
                    arrowcolor=cor,    # mesma cor da linha (vermelha ou branca)
                    opacity=0.95
                )


                # ==========================
        # ==========================
        # 5.2 Construção do frame
        # ==========================
        frame_data_plot = [
            poly,
            go.Scatter(x=df_A["x_norm"], y=df_A["y_norm"], mode="markers",
                       marker=dict(color="blue", size=9, line=dict(color="white", width=1))),
            go.Scatter(x=df_B["x_norm"], y=df_B["y_norm"], mode="markers",
                       marker=dict(color="red", size=9, line=dict(color="white", width=1))),
            go.Scatter(x=[xb] if xb is not None else [], y=[yb] if yb is not None else [],
                       mode="markers", marker=dict(color="white", size=10, line=dict(color="black", width=1))),
            linha_passe  # trace 4
        ]

        # ==========================
        # 5.2.1 Nome do jogador (legenda inferior)
        # ==========================
        annotations = []

        if nome:
            # Equipa com posse
            equipa_posse = pos_nome if pos_nome is not None else "Unknown team"

            # Linha 1 — equipa com posse
            legenda_equipa = dict(
                x=0.2, y=-0.09,  # posição superior (ligeiramente acima da segunda)
                xref="paper", yref="paper",
                text=f"Team in possession: {equipa_posse}",
                showarrow=False,
                font=dict(color="white", size=13),
                align="left",
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor="white",
                borderwidth=1,
                borderpad=4
            )

            # Linha 2 — jogador
            legenda_jogador = dict(
                x=0.2, y=-0.16,  # posição ligeiramente mais abaixo
                xref="paper", yref="paper",
                text=f"Player in possession: {nome}",
                showarrow=False,
                font=dict(color="white", size=13),
                align="left",
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor="white",
                borderwidth=1,
                borderpad=4
            )

            annotations.extend([legenda_equipa, legenda_jogador])

        # ==========================
        # 5.2.2 Título e métricas
        # ==========================
        import pandas as pd

        if period == 1:
            period_text = "1st Half"
        elif period == 2:
            period_text = "2nd Half"
        else:
            period_text = f"Period {period}"

        titulo_texto = f"Match Period: {period}"

        metricas_frame = frame_data.drop_duplicates(subset=["equipa_jogador"])[
            ["equipa_jogador", "stretch_index", "compactacao"]
        ]

        def format_metricas(equipa_nome):
            linha = metricas_frame[metricas_frame["equipa_jogador"] == equipa_nome]
            if linha.empty:
                return "Sem dados"
            l = linha.iloc[0]
            f = lambda v: f"{v:.1f}" if pd.notnull(v) else "—"
            return f"Stretch Index: {f(l.get('stretch_index'))}<br>Compactness: {f(l.get('compactacao'))}"

        metricas_esquerda_txt = f"{equipa_azul}<br><br>{format_metricas(equipa_azul)}"
        metricas_direita_txt = f"{equipa_vermelha}<br><br>{format_metricas(equipa_vermelha)}"

        # fora do campo, fixas no topo (paper coords)
        annotations_metricas = [
            dict(x=0.02, y=0.90, text=metricas_esquerda_txt, showarrow=False,
                 xref="paper", yref="paper", font=dict(color="blue", size=12), align="left"),
            dict(x=0.99, y=0.90, text=metricas_direita_txt, showarrow=False,
                 xref="paper", yref="paper", font=dict(color="red", size=12), align="left")
        ]

        # ==========================
        # 5.2.2.1 Título superior (nomes das equipas)
        # ==========================
        titulo_equipas = dict(
            x=0.5, y=1.12,  # centrado e acima do campo
            xref="paper", yref="paper",
            text=f"<b>{equipa_azul}</b> vs <b>{equipa_vermelha}</b>",
            showarrow=False,
            font=dict(color="black", size=20),
            align="center"
        )

        # ==========================
        # 5.2.3 Criação do frame
        # ==========================
        frames.append(go.Frame(
            name=fmt_ts(ts),
            data=frame_data_plot,
            layout=go.Layout(
                annotations=annotations + annotations_metricas + [arrow_annotation, titulo_equipas]
            )
        ))

        # ==========================
        # 5.2.4 Atualizar slider
        # ==========================
        slider_steps.append({
            "args": [[fmt_ts(ts)], {"frame": {"duration": 0, "redraw": True},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
            "label": fmt_ts(ts),
            "method": "animate"
        })


        # ==========================
    # 5.3 Layout e controlos
    # ==========================
    sliders = [{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"color": "black"}, "prefix": "", "visible": True},
        "pad": {"b": 10, "t": 35},
        "len": 0.9, "x": 0.05, "y": -0.06,
        "steps": slider_steps
    }]

    # Botões de controlo Play / Pause
    play_pause_buttons = [
        {
            "type": "buttons",
            "x": 0.02, "y": 1.12,
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": 350, "redraw": True},
                                 "fromcurrent": True}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate"}]}
            ]
        }
    ]

        # ==========================
    # 5.3 Layout e controlos
    # ==========================
    sliders = [{
        "active": 0,
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"color": "black"}, "prefix": "", "visible": True},
        "pad": {"b": 10, "t": 35},
        "len": 0.9, "x": 0.05, "y": -0.06,
        "steps": slider_steps
    }]

    # ==========================
# 5.5 Layout final
# ==========================
fig.update_layout(
    xaxis=dict(range=[-1, 121], showgrid=False, zeroline=False, ticks="", showticklabels=False, showline=False),
    yaxis=dict(range=[-3, 83], showgrid=False, zeroline=False, ticks="", showticklabels=False, showline=False,
               scaleanchor="x", scaleratio=1),
    plot_bgcolor="#d3d3d3", paper_bgcolor="#d3d3d3",
    autosize=True, width=None, height=None,
    margin=dict(l=0, r=0, t=0, b=0),
    title=None,
    sliders=sliders,
    updatemenus=play_pause_buttons
)

if frames:
    fig.frames = frames

    # ==========================
    # 5.6 Título inicial fixo
    # ==========================
    fig.add_annotation(
        x=0.5, y=1.05,
        xref="paper", yref="paper",
        text=f"<b>{equipa_azul}</b> vs <b>{equipa_vermelha}</b>",
        showarrow=False,
        font=dict(color="black", size=20),
        align="center"
    )

    # ==========================
    # 5.7 Exibição fullscreen real
    # ==========================
    import json, webbrowser

    # força layout flexível antes de exportar
    fig.update_layout(
        autosize=True,
        width=None,
        height=None,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    # converte a figura em JSON
    plot_json = fig.to_json()

    # HTML completo com CSS responsivo
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>{equipa_azul} vs {equipa_vermelha}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
    html, body {{
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background-color: #d3d3d3;
    }}
    #plot {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    }}
    </style>
    </head>
    <body>
    <div id="plot"></div>
    <script>
    var plot_data = {plot_json};
    Plotly.newPlot('plot', plot_data.data, plot_data.layout, {{
    responsive: true,
    displayModeBar: true
    }});
    window.onresize = function() {{
    Plotly.Plots.resize('plot');
    }};
    </script>
    </body>
    </html>
    """

    # grava e abre no browser
    with open("animacao_full.html", "w", encoding="utf-8") as f:
        f.write(html_template)

    webbrowser.open("animacao_full.html")





