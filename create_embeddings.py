import torch
import open_clip
import numpy as np

from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA

model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-H-14')

# Use dataset/images as the image directory
dataset_path = Path("dataset_mini")
images_path = dataset_path / "images"

# Get sorted list of image paths
image_files = sorted(images_path.glob("*.jpg"))

images = []
for path in image_files:
    image = preprocess(Image.open(path)).unsqueeze(0)
    images.append(image)

images_tensor = torch.cat(images)

with torch.no_grad():
    image_features = model.encode_image(images_tensor)

image_features /= image_features.norm(dim=-1, keepdim=True)

pca = PCA(n_components=3)
image_features_3d = pca.fit_transform(image_features.cpu().numpy())


np.save("embeddings/dog_cat.npy", image_features_3d)