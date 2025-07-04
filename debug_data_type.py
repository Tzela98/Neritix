import os
import pyarrow as pa
from PIL import Image
import io
import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.pyplot as plt

def visualize_image_and_mask(data, index=0, alpha=0.5):
    """Visualize image and mask overlay from Arrow table"""
    if not isinstance(data, pa.Table):
        print("Data is not an Arrow table")
        return

    # Extract image data
    img_dict = data.column('image')[index].as_py()
    if not isinstance(img_dict, dict) or 'bytes' not in img_dict:
        print("Unexpected image format:", type(img_dict))
        return
    
    # Load image
    img = Image.open(io.BytesIO(img_dict['bytes']))
    img_arr = np.array(img)
    
    # Extract mask data
    mask_data = data.column('label')[index].as_py()
    mask_arr = None
    
    if isinstance(mask_data, dict) and 'bytes' in mask_data:
        # Mask stored as image bytes
        print('Mask stored as image bytes')
        mask = Image.open(io.BytesIO(mask_data['bytes']))
        mask_arr = np.array(mask)
    elif isinstance(mask_data, bytes):
        # Raw mask bytes
        print('Mask stored as raw bytes')
        mask = Image.open(io.BytesIO(mask_data))
        mask_arr = np.array(mask)
    elif isinstance(mask_data, list):
        # Mask stored as array data
        print('Mask stored as array data')
        mask_arr = np.array(mask_data)
        if mask_arr.ndim == 1:
            # Reshape 1D array to 2D (assuming square image)
            size = int(np.sqrt(len(mask_arr)))
            mask_arr = mask_arr.reshape((size, size))
    
    if mask_arr is None:
        print("Unsupported mask format:", type(mask_data))
        return
    
    # Create overlay
    plt.figure(figsize=(12, 6))
    
    # Plot original image
    plt.subplot(131)
    plt.imshow(img_arr)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot mask
    plt.subplot(132)
    plt.imshow(mask_arr, cmap='jet')
    plt.title('Mask')
    plt.axis('off')
    
    # Plot overlay
    plt.subplot(133)
    plt.imshow(img_arr)
    plt.imshow(mask_arr, cmap='jet', alpha=alpha)
    plt.title('Image + Mask Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def debug_arrow_file(file_path):
    """Comprehensive debugger for arrow files"""
    print(f"\n=== Debugging {os.path.basename(file_path)} ===")
    
    # 1. Verify file exists and is not empty
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size} bytes")
    if file_size == 0:
        raise ValueError("File is empty")

    # 2. Check first 16 bytes for magic numbers
    with open(file_path, 'rb') as f:
        header = f.read(16)
    print(f"Header bytes: {header}")

    # 3. Try different reading methods
    try:
        # Method 1: Standard Arrow
        with pa.memory_map(file_path, 'r') as source:
            try:
                reader = pa.ipc.RecordBatchFileReader(source)
                table = reader.read_all()
                print("Successfully read as Arrow file!")
                return table
            except pa.ArrowInvalid:
                source.seek(0)
                try:
                    reader = pa.ipc.RecordBatchStreamReader(source)
                    table = reader.read_all()
                    print("Successfully read as Arrow stream!")
                    return table
                except Exception as e:
                    print(f"Arrow stream failed: {e}")

        # Method 2: HuggingFace datasets format
        try:
            from datasets import Dataset
            dataset = Dataset.from_file(file_path)
            print("Successfully read as HuggingFace dataset!")
            return dataset
        except ImportError:
            print("HuggingFace datasets package not available")
        except Exception as e:
            print(f"HuggingFace read failed: {e}")

        # Method 3: JSON/other formats
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            print("File appears to be JSON format")
            return data
        except:
            print("Not a JSON file")

        raise ValueError("All read methods failed")
        
    except Exception as e:
        print(f"Final error: {e}")
        raise

def view_image(data, index=0):
    """Try to display an image from the data"""
    if isinstance(data, pa.Table):
        print("\nArrow table columns:", data.column_names)
        img_col = next((col for col in data.column_names if 'image' in col.lower()), None)
        if img_col:
            img_data = data.column(img_col)[index].as_py()
            if isinstance(img_data, bytes):
                Image.open(io.BytesIO(img_data)).show()
            else:
                print(f"Unexpected image format: {type(img_data)}")
    
    elif hasattr(data, '__getitem__'):  # HuggingFace dataset
        sample = data[index]
        if 'image' in sample:
            sample['image'].show()
        else:
            print("Available keys:", list(sample.keys()))
    
    elif isinstance(data, dict):
        print("Dictionary keys:", data.keys())
    
    else:
        print("Unsupported data format")

# Usage
file_path = "coralscapes_data/data-00004-of-00009.arrow"
data = debug_arrow_file(file_path)
visualize_image_and_mask(data, index=0, alpha=0.5)