import os
import shutil
import pandas as pd

# Configuration
image_folder = "coral_dataset_test/images/images"  # Folder containing all images
csv_file = "coral_dataset_test/combined_annotations_remapped.csv"  # CSV with image-class mapping
output_folder = "coral_dataset_test/acropora_only"  # Where to save Acropora images
class_name = "acropora"  # The class you want to filter

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read CSV file
df = pd.read_csv(csv_file)

# Filter for Acropora images
acropora_images = df[df['Label'] == class_name]['Name'].tolist()  # Adjust column names as needed

# Copy matching images
for img_name in acropora_images:
    src_path = os.path.join(image_folder, img_name)
    dst_path = os.path.join(output_folder, img_name)
    
    if os.path.exists(src_path):
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {img_name}")
    else:
        print(f"Missing: {img_name}")

print(f"\nDone! Copied {len(acropora_images)} images to {output_folder}")