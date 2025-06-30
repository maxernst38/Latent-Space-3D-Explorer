import pandas as pd
import plotly.graph_objects as go
from convex_hull import add_convex_hulls

def neutral_color():
    return "#ffbfc0"  # a neutral gray

def highlight_colors():
    # Extended colorblind-friendly palette (20+ colors)
    return [
        "#e74c3c", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#d62728", "#17becf", "#bcbd22", "#7f7f7f",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#5254a3", "#8ca252", "#bd9e39", "#ad494a", "#a55194"
    ]

def compute_centroids(df):
    centroids = []
    for class_name in sorted(set.union(*df["class_names"])):
        indices = [i for i, names in enumerate(df["class_names"]) if class_name in names]
        if indices:
            x_mean = df.loc[indices, "x"].mean()
            y_mean = df.loc[indices, "y"].mean()
            z_mean = df.loc[indices, "z"].mean()
            centroids.append({
                "class_name": class_name,
                "x": x_mean,
                "y": y_mean,
                "z": z_mean,
                "indices": indices
            })
    return pd.DataFrame(centroids)

def plot_embeddings_and_centroids(df, centroids_df, highlight_class_names=None, show_centroids=True, show_lines=True, show_hulls=False):
    if highlight_class_names is None:
        highlight_class_names = []
    color_map = {}
    palette = highlight_colors()
    for idx, class_name in enumerate(highlight_class_names):
        color_map[class_name] = palette[idx % len(palette)]
    marker_colors = [neutral_color()] * len(df)
    marker_size = [3] * len(df)
    centroid_marker_colors = [neutral_color()] * len(centroids_df)
    centroid_marker_size = [7] * len(centroids_df)
    lines = []
    for class_name, color in color_map.items():
        for i, names in enumerate(df["class_names"]):
            if class_name in names:
                marker_colors[i] = color
                marker_size[i] = 5
        for i, row in centroids_df.iterrows():
            if row["class_name"] == class_name:
                centroid_marker_colors[i] = color
                centroid_marker_size[i] = 12
                if show_lines:
                    for idx in row["indices"]:
                        lines.append({
                            "x": [row["x"], df.loc[idx, "x"]],
                            "y": [row["y"], df.loc[idx, "y"]],
                            "z": [row["z"], df.loc[idx, "z"]],
                            "color": color
                        })
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df["x"], y=df["y"], z=df["z"],
        mode="markers",
        marker=dict(size=marker_size, color=marker_colors),
        name="Embeddings",
        text=df["image"],
        hoverinfo="text"
    ))
    if show_centroids:
        fig.add_trace(go.Scatter3d(
            x=centroids_df["x"], y=centroids_df["y"], z=centroids_df["z"],
            mode="markers+text",
            marker=dict(size=centroid_marker_size, color=centroid_marker_colors, symbol="diamond"),
            text=centroids_df["class_name"],
            textposition="top center",
            name="Centroids",
            hoverinfo="text"
        ))
    if show_lines:
        for line in lines:
            fig.add_trace(go.Scatter3d(
                x=line["x"], y=line["y"], z=line["z"],
                mode="lines",
                line=dict(color=line["color"], width=2),
                showlegend=False
            ))
    if show_hulls:
        fig = add_convex_hulls(fig, df, highlight_class_names, color_map)
    fig.update_layout(height=1000, width=1600, title="Embeddings and Class Centroids")
    return fig
