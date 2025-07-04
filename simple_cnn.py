import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import v2 as transforms
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import pyarrow as pa
from PIL import Image

import io
import time
import glob
import logging
from tqdm import tqdm
from datetime import datetime

import matplotlib.pyplot as plt


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coral_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


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


class CoralDataset(Dataset):
    def __init__(self, arrow_paths, transform=None):
        self.transform = transform
        self.samples = []
        
        for path in arrow_paths:
            with open(path, 'rb') as f:
                reader = pa.ipc.RecordBatchStreamReader(f)
                table = reader.read_all()
                
                image_struct = table['image'].combine_chunks()
                label_struct = table['label'].combine_chunks()
                
                for i in range(len(table)):
                    self.samples.append({
                        'image_bytes': image_struct.field('bytes')[i].as_py(),
                        'label_bytes': label_struct.field('bytes')[i].as_py(),
                        'image_path': image_struct.field('path')[i].as_py()
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = np.array(Image.open(io.BytesIO(sample['image_bytes'])))
        mask = np.array(Image.open(io.BytesIO(sample['label_bytes'])))
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Convert mask to instance format needed by Mask R-CNN
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]  # Remove background
        
        masks = mask == obj_ids[:, None, None]
        boxes = [self._get_bbox(m) for m in masks]
        labels = torch.tensor(obj_ids, dtype=torch.int64)
        
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": labels,
            "masks": torch.as_tensor(masks, dtype=torch.uint8)
        }
        
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, target
    
    def _get_bbox(self, mask):
        pos = np.where(mask)
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])
        return [xmin, ymin, xmax, ymax]

    @classmethod
    def create_splits(cls, all_paths, val_ratio=0.2):
        """Create stratified train/validation splits"""
        full_dataset = cls(all_paths, transform=None)
        
        # Get class distribution for stratification
        class_labels = []
        for _, target in full_dataset:
            unique_classes = torch.unique(target['labels'])
            class_labels.append(unique_classes[0].item())  # Use first class per instance
        
        # Verify equal lengths
        assert len(full_dataset) == len(class_labels), \
            f"Mismatched lengths: {len(full_dataset)} samples vs {len(class_labels)} labels"
        
        # Stratified split
        train_idx, val_idx = train_test_split(
            range(len(full_dataset)),
            test_size=val_ratio,
            stratify=class_labels,
            random_state=42
        )
        
        # Create subsets
        train_set = Subset(full_dataset, train_idx)
        val_set = Subset(full_dataset, val_idx)
        
        # Apply transforms
        train_set.dataset.transform = get_transform(train=True)
        val_set.dataset.transform = get_transform(train=False)
        
        # Log distribution
        print(f"\nCreated splits:")
        print(f"Train samples: {len(train_set)}")
        print(f"Val samples: {len(val_set)}")
        
        return train_set, val_set
    
    def get_class_ids(self):
        return [np.unique(mask) for _, mask in self]

def get_transform(train):
    if train:
        return A.Compose([
            A.Resize(800, 800),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(800, 800),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def collate_fn(batch):
    return tuple(zip(*batch))

class CoralMaskRCNN(torch.nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.model = maskrcnn_resnet50_fpn(
            pretrained=True,
            num_classes=num_classes,
            min_size=800,
            max_size=1333
        )
        # Modify for coral details
        self.model.backbone.body.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)

def train_one_epoch(model, optimizer, loader, device, scaler, epoch):
    model.train()
    total_loss = 0
    batch_iter = 0
    
    # Create progress bar
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Train]', unit='batch')
    
    for images, targets in pbar:
        # Data transfer
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        optimizer.zero_grad()
        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        total_loss += losses.item()
        batch_iter += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.item():.4f}',
            'avg_loss': f'{total_loss/batch_iter:.4f}'
        })
    
    avg_loss = total_loss / len(loader)
    logger.info(f'Train Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}')
    return avg_loss

def validate(model, loader, device, epoch):
    model.eval()
    val_loss = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Val]', unit='batch')
    
    with torch.no_grad():
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            
            val_loss += losses.item()
            pbar.set_postfix({
                'val_loss': f'{losses.item():.4f}',
                'avg_val_loss': f'{val_loss/(pbar.n+1):.4f}'
            })
    
    avg_val_loss = val_loss / len(loader)
    logger.info(f'Validation Epoch {epoch+1} - Avg Loss: {avg_val_loss:.4f}')
    return avg_val_loss


def main():

    start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logger.info(f'Starting training at {start_time}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 1. Get all data files
    logger.info('Loading datasets...')
    all_paths = sorted(glob.glob('coralscapes_data/*-of-00009.arrow'))
    
    # 2. Create proper splits
    train_dataset, val_dataset = CoralDataset.create_splits(
        all_paths=all_paths,
        val_ratio=0.2  # 20% validation
    )
    
    # 3. Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # 4. Model setup
    logger.info('Initializing model...')
    model = CoralMaskRCNN(num_classes=40).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler()
    
    # 5. Training loop
    num_epochs = 10
    logger.info(f'Beginning training for {num_epochs} epochs')

    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
        val_loss = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        logger.info(f'Epoch {epoch+1} completed in {epoch_time:.2f}s')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_{start_time}.pth')
            logger.info('New best model saved!')

    logger.info('Training completed!')
    logger.info(f'Best validation loss: {best_val_loss:.4f}')


if __name__ == "__main__":
    main()
