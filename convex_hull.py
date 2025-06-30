import numpy as np
from scipy.spatial import ConvexHull
import plotly.graph_objects as go

def add_convex_hulls(fig, df, highlight_class_names, color_map, alpha=0.5):
    """
    Adds convex hull Mesh3d traces to the given Plotly figure for each highlighted class.
    """
    for class_name in highlight_class_names:
        color = color_map[class_name]
        indices = [i for i, names in enumerate(df["class_names"]) if class_name in names]
        if len(indices) < 4:
            continue  # Need at least 4 points for a 3D hull
        points = df.loc[indices, ["x", "y", "z"]].values
        try:
            hull = ConvexHull(points)
        except Exception:
            continue
        # Convert hex color to rgba for transparency
        color_rgba = f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},{alpha})'
        fig.add_trace(go.Mesh3d(
            x=points[:,0], y=points[:,1], z=points[:,2],
            i=hull.simplices[:,0], j=hull.simplices[:,1], k=hull.simplices[:,2],
            color=color_rgba,
            opacity=alpha,
            name=f"{class_name} hull",
            showscale=False
        ))
    return fig
