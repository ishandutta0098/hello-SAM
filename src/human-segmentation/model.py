import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from segment_anything import SamPredictor, sam_model_registry
from torch.nn import MSELoss
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
import torch
from torchvision import transforms
import numpy as np

from data_module import HumanSegmentationDataset

class SegmentationModel(pl.LightningModule):
    def __init__(self, model_type, checkpoint):
        super(SegmentationModel, self).__init__()
        self.model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.model.train()
        self.loss_fn = MSELoss()
        

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks, original_image_sizes, input_sizes = batch
        images = images
        masks = masks

        # Compute image embeddings
        with torch.no_grad():
            image_embedding = self.model.image_encoder(images)

        # Compute prompt embeddings
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None,
            boxes=None,
            masks=masks
        )

        # Compute low-res masks and IOU predictions
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale masks
        upscaled_masks = self.model.postprocess_masks(low_res_masks, input_sizes, original_image_sizes)
        binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

        # Compute loss
        loss = self.loss_fn(binary_mask, masks)
        self.log('train_loss', loss)
        return loss


    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=1e-4)

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = HumanSegmentationDataset(csv_file='/Users/ishandutta/Documents/personal/hello-SAM/input/segmentation_full_body_mads_dataset_1192_img/df.csv', transform=transform, model=self.model)
        return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)