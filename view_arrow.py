import pyarrow as pa
import numpy as np
from PIL import Image
import io
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

class_mapping = {
    0: "Unlabeled",  # Optional for unlabeled pixels
    1: "seagrass",
    2: "trash",
    3: "other coral dead",
    4: "other coral bleached",
    5: "sand",
    6: "other coral alive",
    7: "human",
    8: "transect tools",
    9: "fish",
    10: "algae covered substrate",
    11: "other animal",
    12: "unknown hard substrate",
    13: "background",
    14: "dark",
    15: "transect line",
    16: "massive/meandering bleached",
    17: "massive/meandering alive",
    18: "rubble",
    19: "branching bleached",
    20: "branching dead",
    21: "millepora",
    22: "branching alive",
    23: "massive/meandering dead",
    24: "clam",
    25: "acropora alive",
    26: "sea cucumber",
    27: "turbinaria",
    28: "table acropora alive",
    29: "sponge",
    30: "anemone",
    31: "pocillopora alive",
    32: "table acropora dead",
    33: "meandering bleached",
    34: "stylophora alive",
    35: "sea urchin",
    36: "meandering alive",
    37: "meandering dead",
    38: "crown of thorn",
    39: "dead clam"
}


with open('coralscapes_data/data-00000-of-00009.arrow', 'rb') as source:
    reader = pa.ipc.RecordBatchStreamReader(source)
    table = reader.read_all()
    
    # Correct way to access nested fields:
    # First get the struct array, then access its fields
    image_struct = table['image'].combine_chunks()
    label_struct = table['label'].combine_chunks()
    
    # Now access the bytes and path fields within each struct
    image_bytes = image_struct.field('bytes')[0].as_py()
    image_path = image_struct.field('path')[0].as_py()
    
    label_bytes = label_struct.field('bytes')[0].as_py()
    label_path = label_struct.field('path')[0].as_py()
    
    print(f"Image path: {image_path}")
    print(f"Label path: {label_path}")
    
    # Load the images
    coral_image = Image.open(io.BytesIO(image_bytes))
    coral_mask = Image.open(io.BytesIO(label_bytes))
    
    # Convert to arrays
    img_array = np.array(coral_image)
    mask_array = np.array(coral_mask)
    unique_classes = np.unique(mask_array)
    
    # Create colormap and legend
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(class_mapping)))
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.Normalize(vmin=0, vmax=len(class_mapping)-1)
    
    # Create legend only for present classes
    legend_patches = [
        mpatches.Patch(color=cmap(norm(cls)), 
                      label=f'{cls}: {class_mapping[cls]}')
        for cls in unique_classes if cls in class_mapping
    ]
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Show original image
    plt.imshow(img_array)
    
    # Overlay mask with transparency
    plt.imshow(mask_array, cmap=cmap, norm=norm, alpha=0.5)
    
    # Add legend
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Coral Image with Segmentation Overlay')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

