import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import OrientationDataset
from orientation_model import OrientationModel



if __name__ == "__main__":
    # Define Albumentations transforms for higher-resolution data augmentation
    transform = A.Compose([
        A.Resize(224, 224),
        A.RandomBrightnessContrast(p=0.2),  # Randomly change brightness and contrast
        A.Rotate(limit=15, p=0.5),  # Rotate image by up to 15 degrees
        A.Normalize(), 
        ToTensorV2()  # Convert the image to PyTorch Tensor
    ])

    # Load data and prepare dataloaders
    assets_dir = './merged_real_clinics_assets/'
    train_dataset = OrientationDataset(assets_dir, split_csv='split_csvs/train.csv', transform=transform)
    val_dataset = OrientationDataset(assets_dir, split_csv='split_csvs/val.csv', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=16)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)

    # Initialize the model
    model = OrientationModel(num_classes=4, lr=1e-3, )

    # Initialize checkpoint callback to save the best model
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    # Initialize a PyTorch Lightning trainer
    trainer = pl.Trainer(max_epochs=4, accelerator='cuda', callbacks=[checkpoint_callback], log_every_n_steps=5)

    # Train the model
    trainer.fit(model, train_loader, val_loader)
    