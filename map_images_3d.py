import streamlit as st
import plotly.express as px

import torch
import open_clip
import matplotlib
import pandas as pd
import numpy as np

from pathlib import Path

from PIL import Image
from sklearn.decomposition import PCA

# Use dataset/images as the image directory
dataset_path = Path("dataset_mini")
images_path = dataset_path / "images"

# Get sorted list of image paths
image_files = sorted(images_path.glob("*.jpg"))

# Read labels from dataset/labels.txt
labels_path = dataset_path / "labels.txt"
classes = []
with open(labels_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            classes.append(parts[1])  # Get the class label

image_features_3d = np.load("embeddings/dog_cat.npy")

# Assign a color to each class
distinct_classes = sorted(set(classes))
# Use a qualitative colormap with high contrast for distinct colors
cmap = matplotlib.colormaps['tab10']  # or 'Set1', 'Set2', etc.
class_to_color = {cls: cmap(i) for i, cls in enumerate(distinct_classes)}
colors = [class_to_color[cls] for cls in classes]

# Prepare DataFrame for Plotly
df = pd.DataFrame(image_features_3d, columns=["x", "y", "z"])
df["label"] = classes
df["image_path"] = [str(p.name) for p in image_files]

# Sidebar filter
label_filter = st.sidebar.multiselect("Filter by class", sorted(set(classes)))
if label_filter:
    df = df[df["label"].isin(label_filter)]

# Plotly 3D scatter
fig = px.scatter_3d(
    df, x="x", y="y", z="z",
    color="label",
    hover_name="image_path",
    height=700
)

st.plotly_chart(fig)

# Optional: show image preview when selected
selected_img = st.selectbox("Preview an image", df["image_path"].tolist())
if selected_img:
    st.image(str(images_path / selected_img), caption=selected_img, use_column_width=True)