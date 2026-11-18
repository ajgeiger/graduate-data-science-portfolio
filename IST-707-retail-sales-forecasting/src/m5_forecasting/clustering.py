"""Keeps track of k-means scores while trying different numbers of clusters."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px


class ClusterScoreTracker:
    """Records the score for each cluster count and plots the elbow curve."""

    def __init__(self):
        self.scores = pd.DataFrame()

    def add_result(self, model, data: np.ndarray, n_clusters: int) -> None:
        result = pd.DataFrame({"score": [model.score(data)], "clusters": [n_clusters]})
        self.scores = pd.concat([result, self.scores], sort=False).reset_index(drop=True)

    def plot(self) -> None:
        figure = px.line(
            self.scores, x="clusters", y="score", title="Clustering Experiment Analysis"
        )
        figure.update_layout(autosize=False, width=1000, height=400)
        figure.show()
