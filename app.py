import streamlit as st
import torch
import open_clip
import numpy as np
import pandas as pd

from PIL import Image
from pathlib import Path
from sklearn.decomposition import PCA
import plotly.express as px

### ─────────────────────────────────────────────
### Load model only once
@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    return model, preprocess

model, preprocess = load_model()


### ─────────────────────────────────────────────
### Dataset / Embedding Selection
datasets_dir = Path("datasets")
embeddings_dir = Path("embeddings")
datasets = [f.name for f in datasets_dir.iterdir() if f.is_dir()]
embeddings = sorted([f.name for f in embeddings_dir.glob("*.npy")])

selection_mode = st.radio("Choose mode", ["Use precomputed embedding", "Generate embedding from dataset"])

### ─────────────────────────────────────────────
### Load Embedding
if selection_mode == "Use precomputed embedding":
    selected_embedding = st.selectbox("Select embedding", embeddings)
    embedding_path = embeddings_dir / selected_embedding
    image_features_3d = np.load(embedding_path)
    # Try to infer dataset name for image loading
    dataset_name = selected_embedding.replace("_pca.npy", "")
    print(dataset_name)
    image_paths = sorted((datasets_dir / dataset_name / "images").glob("*.jpg"))
    image_names = [p.name for p in image_paths]

else:
    selected_dataset = st.selectbox("Select dataset", datasets)
    dataset_path = datasets_dir / selected_dataset
    image_paths = sorted((dataset_path / "images").glob("*.jpg"))
    image_names = [p.name for p in image_paths]

    if st.button("Generate embeddings"):
        images = []
        progress = st.progress(0)
        for i, path in enumerate(image_paths):
            image = preprocess(Image.open(path)).unsqueeze(0)
            images.append(image)
            progress.progress((i + 1) / len(image_paths))

        images_tensor = torch.cat(images)

        with torch.no_grad():
            image_features = model.encode_image(images_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        image_features_3d = PCA(n_components=3).fit_transform(image_features.cpu().numpy())

        save_path = embeddings_dir / f"{selected_dataset}_pca.npy"
        save_path.parent.mkdir(exist_ok=True, parents=True)
        np.save(save_path, image_features_3d)

        st.success(f"Saved embedding to {save_path}")

### ─────────────────────────────────────────────
### Plot if we have valid data
if 'image_features_3d' in locals():
    df = pd.DataFrame(image_features_3d, columns=["x", "y", "z"])
    df["image"] = image_names
    fig = px.scatter_3d(df, x="x", y="y", z="z", hover_name="image", height=700)
    st.plotly_chart(fig)

    # Optional preview
    selected_img = st.selectbox("Preview image", df["image"])
    if selected_img:
        st.image(str(dataset_path / "images" / selected_img), caption=selected_img, use_container_width=True)
