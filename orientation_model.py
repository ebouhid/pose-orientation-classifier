import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import timm
from torch.optim.lr_scheduler import CyclicLR


class OrientationModel(pl.LightningModule):
    def __init__(self, num_classes=4, lr=1e-3, backbone_name="resnet101"):
        super(OrientationModel, self).__init__()
        self.save_hyperparameters()

        self.backbone = timm.create_model(
            backbone_name, pretrained=True, features_only=True, out_indices=[-1])  # Extract last feature map

        # Get the number of channels in the final feature map
        backbone_out_channels = self.backbone.feature_info[-1]['num_chs']

        # Adaptive pooling to make output independent of input resolution
        # Output: (backbone_out_channels, 1, 1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(backbone_out_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        x = self.backbone(x)[-1]  # Get the last layer's output feature map
        x = self.adaptive_pool(x)  # Apply adaptive pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc_layers(x)  # Apply fully connected layers
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels.argmax(1))
        self.log('train_loss', loss, prog_bar=True)
        train_acc = (torch.softmax(outputs, dim=1).argmax(1)
                     == labels.argmax(1)).float().mean()
        self.log('train_acc', train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, labels.argmax(1))
        self.log('val_loss', loss, prog_bar=False)
        val_acc = (torch.softmax(outputs, dim=1).argmax(1)
                   == labels.argmax(1)).float().mean()
        self.log('val_acc', val_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Define the cyclic learning rate scheduler
        scheduler = {
            'scheduler': CyclicLR(
                optimizer,
                base_lr=1e-4,       # Lower boundary of learning rate
                max_lr=1e-2,        # Upper boundary of learning rate
                step_size_up=2000,  # Number of iterations to increase the learning rate
                mode='triangular2',  # 'triangular2' mode, the amplitude decreases after each cycle
                cycle_momentum=False
            ),
            'interval': 'step',    # CyclicLR should be updated every training step
            'frequency': 1,        # Frequency of scheduler update
            'name': 'cyclic_lr_scheduler'
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
