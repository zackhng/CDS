import numpy as np
from PIL import Image
from typing import Tuple
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

class ImageProcessor:
    def __init__(self, target_size: Tuple[int,int]=(224,224), normalize: bool=True):
        self.target_size = target_size
        self.normalize = normalize

        transforms_list = [Resize(target_size), ToTensor()]
        if normalize:
            transforms_list.append(
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]))
        self.transform = Compose(transforms_list)

    def process(self, image_path):
        img = Image.open(image_path).convert('RGB')
        return self.transform(img)