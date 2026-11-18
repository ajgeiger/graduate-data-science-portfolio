"""Confusion matrix plot with row and column totals, used in the
classification notebook.

Adapted from Wagner Cipriano's pretty confusion matrix example for seaborn
(github.com/wcipriano/pretty-print-confusion-matrix).
"""

from __future__ import annotations

from string import ascii_uppercase

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import QuadMesh
from sklearn.metrics import confusion_matrix

DIAGONAL_COLOR = [0.35, 0.8, 0.55, 1.0]
TOTAL_COLOR = [0.27, 0.30, 0.27, 1.0]
GRAND_TOTAL_COLOR = [0.17, 0.20, 0.17, 1.0]


def plot_confusion_matrix_from_data(
    actual,
    predicted,
    labels: list[str] | None = None,
    font_size: int = 11,
    figsize: tuple[int, int] = (9, 9),
    show_zero_cells: int = 2,
) -> None:
    """Plot the confusion matrix for actual vs predicted labels."""
    if not labels:
        n_classes = len(np.unique(actual))
        labels = [f"class {letter}" for letter in ascii_uppercase[:n_classes]]

    counts = confusion_matrix(actual, predicted)
    matrix = pd.DataFrame(counts, index=labels, columns=labels)
    plot_confusion_matrix(
        matrix, font_size=font_size, figsize=figsize, show_zero_cells=show_zero_cells
    )


def plot_confusion_matrix(
    matrix: pd.DataFrame,
    cmap: str = "Oranges",
    font_size: int = 11,
    line_width: float = 0.5,
    figsize: tuple[int, int] = (8, 8),
    show_zero_cells: int = 0,
) -> None:
    """Predicted classes go on the y axis. show_zero_cells: 0 hides empty
    cells, 1 shows a 0, 2 shows the count and percent."""
    matrix = matrix.T
    add_totals(matrix)

    figure = plt.figure("Conf matrix default", figsize)
    axis = figure.gca()
    axis.cla()

    axis = sns.heatmap(
        matrix,
        annot=True,
        annot_kws={"size": font_size},
        linewidths=line_width,
        ax=axis,
        cbar=False,
        cmap=cmap,
        linecolor="w",
        fmt=".2f",
    )
    axis.set_xticklabels(axis.get_xticklabels(), rotation=45, fontsize=10)
    axis.set_yticklabels(axis.get_yticklabels(), rotation=25, fontsize=10)
    for tick in axis.xaxis.get_major_ticks() + axis.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)

    face_colors = axis.findobj(QuadMesh)[0].get_facecolors()
    values = np.array(matrix.to_records(index=False).tolist())

    texts_to_add = []
    texts_to_remove = []
    for position, text in enumerate(axis.collections[0].axes.texts):
        x, y = np.array(text.get_position()) - [0.5, 0.5]
        new_texts, old_texts = format_cell(
            values, int(y), int(x), text, face_colors, position, font_size, show_zero_cells
        )
        texts_to_add.extend(new_texts)
        texts_to_remove.extend(old_texts)

    for text in texts_to_remove:
        text.remove()
    for text in texts_to_add:
        axis.text(text["x"], text["y"], text["text"], **text["kwargs"])

    axis.set_title("Confusion matrix")
    axis.set_xlabel("Actual")
    axis.set_ylabel("Predicted")
    plt.tight_layout()
    plt.show()


def add_totals(matrix: pd.DataFrame) -> None:
    """Add a total column (sum_lin) and a total row (sum_col)."""
    row_totals = [row.sum() for _, row in matrix.iterrows()]
    column_totals = [matrix[column].sum() for column in matrix.columns]
    matrix["sum_lin"] = row_totals
    column_totals.append(np.sum(row_totals))
    matrix.loc["sum_col"] = column_totals


def format_cell(values, row, column, text, face_colors, position, font_size, show_zero_cells):
    """Set the text and color for one cell. Total cells get the count plus
    the percent correct and percent wrong."""
    texts_to_add = []
    texts_to_remove = []
    cell_value = values[row][column]
    grand_total = values[-1][-1]
    percent_of_total = float(cell_value) / grand_total * 100
    last_index = len(values[:, column]) - 1

    if column == last_index or row == last_index:
        if cell_value != 0:
            if column == last_index and row == last_index:
                correct = sum(values[i][i] for i in range(values.shape[0] - 1))
            elif column == last_index:
                correct = values[row][row]
            else:
                correct = values[column][column]
            percent_correct = float(correct) / cell_value * 100
            percent_wrong = 100 - percent_correct
        else:
            percent_correct = percent_wrong = 0

        correct_label = "100%" if percent_correct == 100 else f"{percent_correct:.2f}%"
        texts_to_remove.append(text)

        font = font_manager.FontProperties(weight="bold", size=font_size)
        base_kwargs = dict(color="w", ha="center", va="center", gid="sum", fontproperties=font)
        labels = [f"{cell_value:d}", correct_label, f"{percent_wrong:.2f}%"]
        colors = ["w", "g", "r"]
        offsets = [-0.3, 0, 0.3]
        for label, color, offset in zip(labels, colors, offsets, strict=True):
            kwargs = {**base_kwargs, "color": color}
            texts_to_add.append(
                {"x": text._x, "y": text._y + offset, "text": label, "kwargs": kwargs}
            )

        is_grand_total = column == last_index and row == last_index
        face_colors[position] = GRAND_TOTAL_COLOR if is_grand_total else TOTAL_COLOR
    else:
        if percent_of_total > 0:
            label = f"{cell_value}\n{percent_of_total:.2f}%"
        elif show_zero_cells == 0:
            label = ""
        elif show_zero_cells == 1:
            label = "0"
        else:
            label = "0\n0.0%"
        text.set_text(label)

        if column == row:
            text.set_color("w")
            face_colors[position] = DIAGONAL_COLOR
        else:
            text.set_color("r")

    return texts_to_add, texts_to_remove
