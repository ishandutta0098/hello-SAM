import pytorch_lightning as pl

from model import SegmentationModel

def main():
    model = SegmentationModel(model_type='vit_b', checkpoint='/Users/ishandutta/Documents/personal/hello-SAM/checkpoints/sam_vit_b_01ec64.pth')
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model)

if __name__ == '__main__':
    main()