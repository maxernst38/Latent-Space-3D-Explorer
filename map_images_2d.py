import torch
import open_clip
import matplotlib

from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-H-14')

# Use dataset/images as the image directory
dataset_path = Path("dataset_mini")
images_path = dataset_path / "images"

# Read labels from dataset/labels.txt
labels_path = dataset_path / "labels.txt"
classes = []
with open(labels_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 2:
            classes.append(parts[1])  # Get the class label

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

pca = PCA(n_components=2)
image_features_2d = pca.fit_transform(image_features.cpu().numpy())

# Assign a color to each class
distinct_classes = sorted(set(classes))
# Use a qualitative colormap with high contrast for distinct colors
cmap = matplotlib.colormaps['tab10']  # or 'Set1', 'Set2', etc.
class_to_color = {cls: cmap(i) for i, cls in enumerate(distinct_classes)}
colors = [class_to_color[cls] for cls in classes]

# Plot in 3D
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(
    image_features_2d[:, 0],
    image_features_2d[:, 1],
    c=colors, marker='o'
)
ax.set_title('2D PCA of CLIP Image Features')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')

# Add legend
for cls in distinct_classes:
    ax.scatter([], [], [], c=[class_to_color[cls]], label=cls)
ax.legend(title='Class')
plt.show()

