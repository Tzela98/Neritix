import pyarrow as pa
import numpy as np
from PIL import Image
import io
import json
import matplotlib.pyplot as plt

class_mapping = {
    0: "Background",
    1: "Hard Substrate",
    2: "Massive/Meandering Alive",
    3: "Sand",
    4: "Algae Covered Substrate",
    5: "Dark",
    6: "Rubble",
    7: "Other Coral Alive",
    8: "Table Acropora Alive",
    9: "Pocillopora Alive",
    10: "Acropora Alive",
    11: "Human",
    12: "Branching Dead",
    13: "Meandering Alive",
    14: "Massive/Meandering Dead",
    15: "Table Acropora Dead",
    16: "Seagrass",
    17: "Fish",
    18: "Branching Alive",
    19: "Branching Bleached",
    20: "Other Coral Dead",
    21: "Millepora",
    22: "Meandering Dead",
    23: "Transect Line",
    24: "Other Coral Bleached",
    25: "Massive/Meandering Bleached",
    26: "Stylophora Alive",
    27: "Sponge",
    28: "Turbinaria",
    29: "Transect Tools",
    30: "Anemone",
    31: "Meandering Bleached",
    32: "Crown of Thorns",
    33: "Clam",
    34: "Trash",
    35: "Sea Urchin",
    36: "Other Animal",
    37: "Sea Cucumber",
    38: "Dead Clam"
}


with open('coralscapes_data/data-00000-of-00009.arrow', 'rb') as source:
    reader = pa.ipc.RecordBatchStreamReader(source)
    table = reader.read_all()
    
    # Correct way to access nested struct fields
    label_bytes = table['label'].combine_chunks().field('bytes')
    label_paths = table['label'].combine_chunks().field('path')
    
    # Get the first label's binary data
    first_label_bytes = label_bytes[0].as_py()
    first_label_path = label_paths[0].as_py()
    
    print(f"First label path: {first_label_path}")
    print(f"First label bytes length: {len(first_label_bytes)}")
    
    # Function to try interpreting the label
    def interpret_label(label_bytes):
        img = Image.open(io.BytesIO(label_bytes))
        print("Label is an image:")
        print(f"Format: {img.format}, Size: {img.size}, Mode: {img.mode}")
        if img.mode in ('L', 'P', '1'):
            arr = np.array(img)
            print(f"Unique values: {np.unique(arr)}")
        return arr
    
    coral_labels = interpret_label(first_label_bytes)

    mask = Image.open(io.BytesIO(first_label_bytes))

    plt.imshow(mask, cmap='nipy_spectral')
    plt.colorbar()
    plt.legend(coral_labels)
    plt.title('Coral Segmentation Mask')
    plt.show()