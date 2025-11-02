# grid_hex.py
# Grelha hexagonal para campos (ex.: 120x80). Desenha linhas num Axes existente.

from typing import List, Tuple
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

__all__ = ["draw_hex_grid", "get_hex_centers", "map_to_hex_cell"]

def _hex_corner(center: Tuple[float, float], size: float, i: int) -> Tuple[float, float]:
    """Coordenada do vértice i (0..5) do hexágono com centro e raio 'size'."""
    angle_rad = np.deg2rad(60 * i)
    return (
        center[0] + size * np.cos(angle_rad),
        center[1] + size * np.sin(angle_rad),
    )

def draw_hex_grid(
    ax: Axes,
    *,
    field_width: float = 120.0,
    field_height: float = 80.0,
    x0: float = 0.0,
    y0: float = 0.0,
    hex_size: float = 1.86,
    color: str = "black",
    linestyle: str = "--",
    linewidth: float = 0.5,
    zorder: int = 1,
) -> List[Line2D]:
    """
    Desenha uma grelha hexagonal no Axes indicado, sem alterar limites/escala.

    Parâmetros:
        ax            : Axes do matplotlib
        field_width   : Largura total do campo (horizontal)
        field_height  : Altura total do campo (vertical)
        x0, y0        : Offset da origem (canto inferior esquerdo)
        hex_size      : Raio de cada hexágono
        color         : Cor das linhas
        linestyle     : Tipo de linha
        linewidth     : Espessura da linha
        zorder        : Ordem de desenho

    Retorna:
        Lista de linhas desenhadas (Line2D)
    """
    dx = 3/2 * hex_size
    dy = np.sqrt(3) * hex_size

    n_cols = int(np.ceil(field_width / dx))
    n_rows = int(np.ceil(field_height / dy))

    lines = []

    for col in range(n_cols):
        for row in range(n_rows):
            cx = x0 + col * dx
            cy = y0 + row * dy + (hex_size * np.sqrt(3)/2 if col % 2 else 0)
            corners = [_hex_corner((cx, cy), hex_size, i) for i in range(6)]
            for i in range(6):
                p1 = corners[i]
                p2 = corners[(i + 1) % 6]
                line = Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder)
                ax.add_line(line)
                lines.append(line)

    return lines

def get_hex_centers(
    field_width: float = 120.0,
    field_height: float = 80.0,
    x0: float = 0.0,
    y0: float = 0.0,
    hex_size: float = 1.86
) -> List[Tuple[float, float]]:
    """Calcula os centros de todas as células hexagonais."""
    dx = 3/2 * hex_size
    dy = np.sqrt(3) * hex_size

    n_cols = int(np.ceil(field_width / dx))
    n_rows = int(np.ceil(field_height / dy))

    centers = []
    for col in range(n_cols):
        for row in range(n_rows):
            cx = x0 + col * dx
            cy = y0 + row * dy + (hex_size * np.sqrt(3)/2 if col % 2 else 0)
            centers.append((cx, cy))

    return centers

def map_to_hex_cell(
    x: float,
    y: float,
    hex_size: float = 1.86,
    x0: float = 0.0,
    y0: float = 0.0
) -> Tuple[int, int]:
    """
    Mapeia coordenadas (x, y) para o índice (coluna, linha) da grelha.
    Retorna (col, row) com origem no canto inferior esquerdo como (0, 0).
    """
    dx = 3/2 * hex_size
    dy = np.sqrt(3) * hex_size

    col = int((x - x0) / dx)
    row_offset = hex_size * np.sqrt(3)/2 if col % 2 else 0
    row = int((y - y0 - row_offset) / dy)

    return col, row

# ==========================
# gerar_hex_grid
# ==========================
import numpy as np

import numpy as np

import numpy as np

def gerar_hex_grid(field_width=120.0, field_height=80.0, hex_size=1.84):
    """
    Gera uma grelha hexagonal perfeitamente ajustada ao campo:
    - Começa no canto inferior esquerdo (0,0)
    - Termina exatamente em (field_width, field_height)
    - Cada hexágono tem área ≈ 9 m² (hex_size=1.86)
    """
    hexes = []
    dx = 3/2 * hex_size
    dy = np.sqrt(3) * hex_size

    # deslocar o centro inicial de forma que o 1º vértice esteja em (0,0)
    x0 = hex_size
    y0 = hex_size * np.sqrt(3) / 2

    # calcular número de colunas e linhas exatas
    n_cols = int(np.floor((field_width - hex_size) / dx)) + 1
    n_rows = int(np.floor((field_height - y0) / dy)) + 1

    for col in range(n_cols):
        for row in range(n_rows):
            cx = x0 + col * dx
            cy = y0 + row * dy + (hex_size * np.sqrt(3)/2 if col % 2 else 0)

            # ignorar hexágonos fora dos limites
            if cx + hex_size > field_width or cy + hex_size > field_height:
                continue

            # calcular vértices do hexágono
            corners = [
                (cx + hex_size * np.cos(np.deg2rad(angle)),
                 cy + hex_size * np.sin(np.deg2rad(angle)))
                for angle in range(0, 360, 60)
            ]
            hexes.append(corners)

    return hexes

