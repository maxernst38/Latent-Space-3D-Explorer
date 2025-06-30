import torch
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

def generate_embeddings(image_paths, preprocess, model, device, n_components=3, batch_size=16):
    n_images = len(image_paths)
    all_features = []
    for start in range(0, n_images, batch_size):
        end = min(start + batch_size, n_images)
        batch_paths = image_paths[start:end]
        batch_images = [preprocess(Image.open(p)).unsqueeze(0).to(device) for p in batch_paths]
        batch_tensor = torch.cat(batch_images)
        with torch.no_grad():
            batch_features = model.encode_image(batch_tensor)
        batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
        all_features.append(batch_features.cpu())
    image_features = torch.cat(all_features, dim=0)
    image_features_3d = PCA(n_components=n_components).fit_transform(image_features.numpy())
    return image_features_3d

def save_embedding(embedding, save_path):
    save_path.parent.mkdir(exist_ok=True, parents=True)
    np.save(save_path, embedding)
