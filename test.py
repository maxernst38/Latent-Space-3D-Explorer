import torch
import open_clip

from PIL import Image

model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-H-14')

image = preprocess(Image.open("images/cat_dog.jpg")).unsqueeze(0)
text = tokenizer(["a cat", "a dog"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

# similarity (normalized)
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

print(image_features.shape, text_features.shape)
print(image_features)
similarity = (image_features @ text_features.T)
print(similarity)

