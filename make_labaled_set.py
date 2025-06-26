import os
import shutil

# Define your source directories
cats_dir = 'test_dataset/cats_set'
dogs_dir = 'test_dataset/dogs_set'
output_dir = 'dataset_mini/images'
labels_file = 'dataset_mini/labels.txt'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Helper to process a set (limit to 10 images)
def process_set(src_dir, label, start_idx, max_images=10):
    entries = []
    count = 0
    for fname in os.listdir(src_dir):
        src_path = os.path.join(src_dir, fname)
        if os.path.isfile(src_path):
            # Copy image to output_dir with unique name
            new_fname = f"img_{start_idx}.jpg"
            dst_path = os.path.join(output_dir, new_fname)
            shutil.copy2(src_path, dst_path)
            # Add label entry
            entries.append(f"{new_fname} {label}")
            start_idx += 1
            count += 1
            if count >= max_images:
                break
    return entries, start_idx

# Process cats
cat_entries, idx = process_set(cats_dir, 'cat', 0, max_images=30)
# Process dogs
dog_entries, _ = process_set(dogs_dir, 'dog', idx, max_images=20)

# Write labels file
with open(labels_file, 'w') as f:
    for entry in cat_entries + dog_entries:
        f.write(entry + '\n')

print(f"Dataset merged in '{output_dir}'. Labels in '{labels_file}'.")