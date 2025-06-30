from pathlib import Path
import numpy as np
import yaml
import pandas as pd

def get_datasets(datasets_dir):
    return [f.name for f in Path(datasets_dir).iterdir() if f.is_dir()]

def get_embeddings(embeddings_dir):
    return sorted([f.name for f in Path(embeddings_dir).glob("*.npy")])

def load_embedding(embedding_path):
    return np.load(embedding_path)

def load_class_names(dataset_yaml):
    if Path(dataset_yaml).exists():
        with open(dataset_yaml) as f:
            data = yaml.safe_load(f)
            return data.get("names", [])
    return []

def load_image_and_label_paths(images_dir, labels_dir, selected_split=None):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    if selected_split:
        image_paths = sorted((images_dir / selected_split).glob("*.jpg"))
        image_names = [p.name for p in image_paths]
        label_split_dir = labels_dir / selected_split
    else:
        image_paths = sorted(images_dir.glob("*.jpg"))
        image_names = [p.name for p in image_paths]
        label_split_dir = labels_dir
    return image_paths, image_names, label_split_dir

def parse_labels(image_names, label_split_dir, class_names):
    image_class_indices = []
    image_class_names = []
    all_class_names = set()
    for img_name in image_names:
        label_file = Path(label_split_dir) / img_name.replace(".jpg", ".txt")
        indices = set()
        names = set()
        if label_file.exists():
            with open(label_file) as f:
                for line in f:
                    if line.strip():
                        class_id = line.split()[0]
                        indices.add(class_id)
                        try:
                            class_idx = int(class_id)
                            if 0 <= class_idx < len(class_names):
                                class_name = class_names[class_idx]
                                names.add(class_name)
                                all_class_names.add(class_name)
                        except Exception:
                            pass
        image_class_indices.append(indices)
        image_class_names.append(names)
    return image_class_indices, image_class_names, all_class_names
