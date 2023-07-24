import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from segment_anything.utils.transforms import ResizeLongestSide
import numpy as np

class HumanSegmentationDataset(Dataset):
    def __init__(self, csv_file, model, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.transform = transform
        self.resize_transform = ResizeLongestSide(model.image_encoder.img_size)
        self.model = model

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.dataframe.loc[idx, 'images']
        img_name = os.path.join("/Users/ishandutta/Documents/personal/hello-SAM/input/segmentation_full_body_mads_dataset_1192_img", img_name)

        mask_name = self.dataframe.loc[idx, 'masks']
        mask_name = os.path.join("/Users/ishandutta/Documents/personal/hello-SAM/input/segmentation_full_body_mads_dataset_1192_img", mask_name)

        image = Image.open(img_name).convert("RGB")
        original_image_size = np.array(image).shape[:2]
        image = self.resize_transform.apply_image(np.array(image))
        image_torch = torch.as_tensor(image)
        transformed_image = image_torch.permute(2, 0, 1).contiguous()[:, :, :]
        input_image = self.model.preprocess(transformed_image)
        input_size = tuple(transformed_image.shape[-2:])

        mask = Image.open(mask_name).convert("L")  # Convert to grayscale
        mask = np.array(mask)
        mask = torch.from_numpy(mask)  # Convert to torch.Tensor
        mask = mask.float()  # Ensure the mask is float type

        return input_image, mask, original_image_size, input_size
