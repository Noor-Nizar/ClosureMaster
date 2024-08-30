import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, processor):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)]
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Convert the image to a numpy array
        image_np = np.array(image)
        
        # Process the image with the processor and return pixel values
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        # Return the numpy array and the pixel values tensor
        return image_np, pixel_values.squeeze(0)  # Remove the batch dimension