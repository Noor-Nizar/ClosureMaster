import os
from PIL import Image
from torch.utils.data import Dataset
import torch

import torchvision.transforms as T

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, processor, test_n_samples=None):
        self.image_dir = image_dir
        self.test_n_samples = test_n_samples
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
        self.processor = processor
        
        
        # Transformations for scales used in the loopnet paper
        # torch.Size([1, 7, 480, 640])
        # torch.Size([1, 7, 240, 320])
        # torch.Size([1, 7, 120, 160])
        self.t1 = T.Resize((480, 640))
        self.t2 = T.Resize((240, 320))
        self.t3 = T.Resize((120, 160))

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
        
        # Apply 1/2 and 1/4 scaling
        s1 = self.t1(full_scale)
        s2 = self.t2(full_scale)
        s3 = self.t3(full_scale)
        return s1, s2, s3