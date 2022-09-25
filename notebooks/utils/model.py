"""The model script."""

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

# The model.
class HHHHBModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        """
        Instantiate the HHHHBModel class based on the pl.LightningModule
        (https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html).
        
        Parameters
        ----------
        arch
            The architecture code.
        encoder_name
            The name of the encoder.
        in_channels
            The number of channels of the input image files.
        out_classes
            The number of classes that define the label mask.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Instantiate datasets, model, and trainer params if provided.
        self.model = self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.learning_rate = self.hparams.get("lr", 1e-3)

    ## Required LightningModule methods ##

    def forward(self, image: torch.Tensor):
        return self.model(image)

    def training_step(self, batch: dict, batch_idx: int):
        """
        Training step.
        
        Parameters
        ----------
        batch
            A dictionary of items from HHHHBDataset of the form
            {'id': list[int], 'image': list[torch.Tensor], 'mask': list[torch.Tensor]]}
        batch_idx
            The batch number.
        """
        # Load images and labels
        x = batch["image"]
        y = batch["mask"].long()
        # Forward pass
        logits = self.forward(x) 
        loss = self.loss_fn(logits, y)
        # Log batch loss.
        self.log("train_score", loss, prog_bar=True) 
        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        """
        Validation step.
        
        Parameters
        ----------
        batch
            A dictionary of items from HHHHBDataset of the form
            {'id': list[int], 'image': list[torch.Tensor], 'mask': list[torch.Tensor]}
        batch_idx
            The batch number
        """
        # Load images and labels
        x = batch["image"]
        y = batch["mask"]
        # Forward pass 
        logits = self.forward(x)
        # Log batch dice.
        val_score = self.loss_fn(logits, y)
        self.log("val_score", val_score, prog_bar=True)
        return val_score

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return opt