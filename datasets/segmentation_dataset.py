import os
from PIL import Image
from torch.utils.data import Dataset
import torch

import torchvision.transforms as T

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, processor, test_n_samples=None):
        # self.image_dir = image_dir
        self.test_n_samples = test_n_samples
        self.image_paths = image_paths
        self.processor = processor
        
        
        # Transformations for scales used in the loopnet paper, including normalization
        self.transforms = {
            'full': T.Compose([
                T.Resize((480, 640)),
                # T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'half': T.Compose([
                T.Resize((240, 320)),
                # T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'quarter': T.Compose([
                T.Resize((120, 160)),
                # T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }

    def __len__(self):
        if self.test_n_samples:
            return min(self.test_n_samples, len(self.image_paths))
        else:
            return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Process the image with the processor and return pixel values
        full_scale = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
        # Apply scaling
        s1 = self.transforms['full'](full_scale)
        s2 = self.transforms['half'](full_scale)
        s3 = self.transforms['quarter'](full_scale)
        return s1, s2, s3