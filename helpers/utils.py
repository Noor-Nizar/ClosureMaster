import numpy as np
from helpers.visualization import ade_palette

from torch import tensor
import torch

def classify_objects(segmentation):
    """
    Classify the objects in the segmentation map into three categories:
    static, weakly dynamic, and dynamic.
    
    Args:
    segmentation: numpy array, segmentation map
    
    Returns:
    classified_map: numpy array, map with pixel values 0, 0.5, 1 for static,
                    weakly dynamic, and dynamic objects respectively
    """
    static_classes = [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,
                       10,  11,       13,  14,  15,  16,  17,  18,  19,
                            21,  22,  23,  24,  25,  26,  27,  28,  29,
                       30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                       40,  41,  42,  43,  44,  45,  46,  47,  48,  49,
                       50,  51,  52,  53,  54,  55,  56,  57,  58,  59,
                       60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
                       70,  71,  72,  73,  74,            77,  78,  79,
                            81,  82,       84,  85,  86,  87,  88,  89,
                            91,  92,  93,  94,  95,  96,  97,  98,  99,
                      100, 101,           104, 105, 106, 107,      109,
                      110, 111, 112, 113, 114, 115,      117, 118,     
                      120, 121, 122, 123, 124, 125,           128, 129,
                      130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
                      140, 141, 142, 143, 144, 145, 146, 147, 148, 149]
    
    # Example: buildings (1), trees (4), walls (0)
    
    weakly_dynamic_classes = [20, 75, 76, 80, 83, 90, 102, 103, 108, 116, 119, 127]
    
    # Example: vehicles, buses, trains
    
    dynamic_classes = [12, 126]
    
    # Example: pedestrians, cyclists, vendors
    
    classified_map = np.zeros_like(segmentation, dtype=np.float32)
    
    # Assign pixel intensities based on classification
    classified_map[np.isin(segmentation, static_classes)] = 0
    classified_map[np.isin(segmentation, weakly_dynamic_classes)] = 0.5
    classified_map[np.isin(segmentation, dynamic_classes)] = 1
    
    return classified_map

def classify_objects_tensor(segmentation):
    """
    Classify the objects in the segmentation map into three categories:
    static, weakly dynamic, and dynamic.
    
    Args:
    segmentation: torch.Tensor, segmentation map
    
    Returns:
    classified_map: torch.Tensor, map with pixel values 0, 0.5, 1 for static,
                    weakly dynamic, and dynamic objects respectively
    """
    static_classes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                   10, 11, 13, 14, 15, 16, 17, 18, 19,
                                   21, 22, 23, 24, 25, 26, 27, 28, 29,
                                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                                   70, 71, 72, 73, 74, 77, 78, 79,
                                   81, 82, 84, 85, 86, 87, 88, 89,
                                   91, 92, 93, 94, 95, 96, 97, 98, 99,
                                   100, 101, 104, 105, 106, 107, 109,
                                   110, 111, 112, 113, 114, 115, 117, 118,
                                   120, 121, 122, 123, 124, 125, 128, 129,
                                   130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
                                   140, 141, 142, 143, 144, 145, 146, 147, 148, 149])
    
    weakly_dynamic_classes = torch.tensor([20, 75, 76, 80, 83, 90, 102, 103, 108, 116, 119, 127])
    
    dynamic_classes = torch.tensor([12, 126])
    
    classified_map = torch.zeros_like(segmentation, dtype=torch.float32)
    
    # Assign pixel intensities based on classification
    classified_map[(segmentation.unsqueeze(-1) == static_classes).any(-1)] = 0
    classified_map[(segmentation.unsqueeze(-1) == weakly_dynamic_classes).any(-1)] = 0.5
    classified_map[(segmentation.unsqueeze(-1) == dynamic_classes).any(-1)] = 1
    
    return classified_map

def classify_objects_tensor_batched(segmentations):
    """
    Classify the objects in multiple segmentation maps into three categories:
    static, weakly dynamic, and dynamic.
    
    Args:
    segmentations: torch.Tensor, batch of segmentation maps (B, H, W)
    
    Returns:
    classified_maps: torch.Tensor, batch of maps with pixel values 0, 0.5, 1 for static,
                     weakly dynamic, and dynamic objects respectively (B, H, W)
    """
    static_classes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                   10, 11, 13, 14, 15, 16, 17, 18, 19,
                                   21, 22, 23, 24, 25, 26, 27, 28, 29,
                                   30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                                   40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                                   50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                                   60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
                                   70, 71, 72, 73, 74, 77, 78, 79,
                                   81, 82, 84, 85, 86, 87, 88, 89,
                                   91, 92, 93, 94, 95, 96, 97, 98, 99,
                                   100, 101, 104, 105, 106, 107, 109,
                                   110, 111, 112, 113, 114, 115, 117, 118,
                                   120, 121, 122, 123, 124, 125, 128, 129,
                                   130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
                                   140, 141, 142, 143, 144, 145, 146, 147, 148, 149], device=segmentations.device)
    
    weakly_dynamic_classes = torch.tensor([20, 75, 76, 80, 83, 90, 102, 103, 108, 116, 119, 127], device=segmentations.device)
    
    dynamic_classes = torch.tensor([12, 126], device=segmentations.device)
    
    classified_maps = torch.zeros_like(segmentations, dtype=torch.float32)
    
    # Assign pixel intensities based on classification
    classified_maps[(segmentations.unsqueeze(-1) == static_classes).any(-1)] = 0
    classified_maps[(segmentations.unsqueeze(-1) == weakly_dynamic_classes).any(-1)] = 0.5
    classified_maps[(segmentations.unsqueeze(-1) == dynamic_classes).any(-1)] = 1
    
    return classified_maps