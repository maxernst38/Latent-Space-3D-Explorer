import streamlit as st
import torch
import open_clip
import numpy as np
import pandas as pd
import yaml

from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
import plotly.express as px
from embedding import generate_embeddings, save_embedding
from data_loading import (
    get_datasets, get_embeddings, load_embedding, load_class_names, load_image_and_label_paths, parse_labels
)
from visualization import compute_centroids, plot_embeddings_and_centroids

### ─────────────────────────────────────────────
### Load model only once
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_e16')

    return model, preprocess

model, preprocess = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

### ─────────────────────────────────────────────
### Dataset / Embedding Selection
datasets_dir = Path("datasets")
embeddings_dir = Path("embeddings")

def main():
    datasets = get_datasets(datasets_dir)
    embeddings = get_embeddings(embeddings_dir)
    selection_mode = st.radio("Choose mode", ["Use precomputed embedding", "Generate embedding from dataset"])
    dataset_format = st.selectbox("Select dataset format", ["YOLO object detection"])

    if selection_mode == "Generate embedding from dataset":
        selected_dataset = st.selectbox("Select dataset", datasets)
        dataset_path = datasets_dir / selected_dataset
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        split_options = [d.name for d in images_dir.iterdir() if d.is_dir()]
        if split_options:
            selected_split = st.selectbox("Select split", split_options)
        else:
            selected_split = None
        image_paths, image_names, label_split_dir = load_image_and_label_paths(images_dir, labels_dir, selected_split)
        if st.button("Generate embeddings"):
            embedding = generate_embeddings(image_paths, preprocess, model, device)
            save_path = embeddings_dir / f"{selected_dataset}_{selected_split}_pca.npy"
            save_embedding(embedding, save_path)
            st.success(f"Saved embedding to {save_path}")
            st.stop()
    elif selection_mode == "Use precomputed embedding":
        selected_embedding = st.selectbox("Select embedding", embeddings)
        embedding_path = embeddings_dir / selected_embedding
        image_features_3d = load_embedding(embedding_path)
        base_name = selected_embedding.replace("_pca.npy", "")
        parts = base_name.rsplit("_", 1)
        if len(parts) == 2:
            dataset_name, selected_split = parts
        else:
            dataset_name = base_name
            selected_split = None
        dataset_path = datasets_dir / dataset_name
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        split_options = [d.name for d in images_dir.iterdir() if d.is_dir()]
        image_paths, image_names, label_split_dir = load_image_and_label_paths(images_dir, labels_dir, selected_split)
        df = pd.DataFrame(image_features_3d, columns=["x", "y", "z"])
        df["image"] = image_names
        class_names = load_class_names(dataset_path / "dataset.yaml")
        image_class_indices, image_class_names, all_class_names = parse_labels(image_names, label_split_dir, class_names)
        df["class_indices"] = image_class_indices
        df["class_names"] = image_class_names
        centroids_df = compute_centroids(df)
        highlight_class_names = st.multiselect(
            "Highlight class(es) (centroid + embeddings)",
            sorted(all_class_names),
            default=[]
        )
        show_centroids = st.checkbox("Show centroids", value=True)
        show_lines = st.checkbox("Show lines to centroid when highlighted", value=True)
        show_hulls = st.checkbox("Show convex hulls for highlighted classes", value=False)
        fig = plot_embeddings_and_centroids(
            df, centroids_df, highlight_class_names,
            show_centroids=show_centroids, show_lines=show_lines, show_hulls=show_hulls
        )
        st.plotly_chart(fig, use_container_width=True)
        # Optional preview
        if split_options:
            selected_img = st.selectbox("Preview image", df["image"])
            if selected_img:
                st.image(str(images_dir / selected_split / selected_img), caption=selected_img, use_container_width=True)
        else:
            selected_img = st.selectbox("Preview image", df["image"])
            if selected_img:
                st.image(str(images_dir / selected_img), caption=selected_img, use_container_width=True)

if __name__ == "__main__":
    main()
