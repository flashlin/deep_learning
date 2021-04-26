import pytorch_lightning as pl
import glob
import os
import time
from pytorch_lightning import loggers as pl_loggers
import torch
from torch import nn as nn
import torch.optim as optim
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np

from ptorch.ImageClassificationDataset import ImageClassification
from py.Info import info


class LitModelTrainer(pl.LightningModule):
    checkpoints_dir = "d:/demo/training/checkpoints"
    max_epochs = 180

    def __init__(self, model):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = model
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # # Example input for visualizing the graph in Tensorboard
        # self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

    def configure_optimizers(self):
        # optimizer = self.model.get_optimizer()
        # if optimizer is None:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        return [optimizer], [scheduler]

    def func_exists(self, class_name, func_name):
        return hasattr(class_name, func_name) and callable(getattr(class_name, func_name))

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        x, labels = batch
        preds = self.model(x)
        #predicted = torch.max(preds, 1)[1]
        loss = self.loss_module(preds, labels)
        assert loss >= 0, f"{loss} should >= 0"
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log('train_acc', acc, on_step=False,
                 on_epoch=True)  # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('test_acc',
                 acc)  # By default logs it per epoch (weighted average over batches), and returns it afterwards

    def predict_image(self, image_path):
        pretrained_filename = self.get_pretrained_filename()
        model = LitModelTrainer.load_from_checkpoint(pretrained_filename)
        model.freeze()
        image_classification = ImageClassification()
        img = image_classification.get_image_data(image_path)
        output = model(img)

        print(f"output = {output}")

        _, preds_tensor = torch.max(output, 1)
        # print(f"{torch.max(output, 1)}")

        preds = np.squeeze(preds_tensor.numpy())

        # print("Predicted:", preds[:10])
        # print(f"{preds}")
        return preds

    def fit_model(self, dataset, save_name=None, **kwargs):
        if save_name is None:
            save_name = self.get_model_name()

        trainer = self.create_trainer(save_name)

        model = self.load_pretrained_model(save_name)

        train_loader, val_loader = dataset.get_train_validation_loaders()
        trainer.fit(model, train_loader, val_loader)

        model, result = self.verify_model_results(trainer, model, val_loader)

        return model, result

    def verify_model_results(self, trainer, model, val_loader):
        model = LitModelTrainer.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training
        # Test best model on validation and test set
        val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
        # test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
        result = {
            # "test": test_result[0]["test_acc"],
            "val": val_result[0]["test_acc"]
        }
        return model, result

    def load_pretrained_model(self, save_name):
        # Check whether pretrained model exists. If yes, load it and skip training
        pretrained_filename = self.get_pretrained_filename(save_name)
        if os.path.isfile(pretrained_filename):
            info("Found pretrained model at %s, loading..." % pretrained_filename)
            model = LitModelTrainer.load_from_checkpoint(pretrained_filename)
        else:
            # pl.seed_everything(42)  # To be reproducable
            info(f"model name = {save_name}")
            model = self
        return model

    def create_trainer(self, save_name):
        # Create a PyTorch Lightning trainer with the generation callback
        cuda_available = torch.cuda.is_available
        trainer = pl.Trainer(default_root_dir=os.path.join(self.checkpoints_dir, save_name),
                             checkpoint_callback=ModelCheckpoint(save_weights_only=True,
                                                                 mode="min",
                                                                 monitor="val_acc",
                                                                 save_last=True),
                             gpus=1 if cuda_available is True else 0,
                             max_epochs=self.max_epochs,
                             callbacks=[LearningRateMonitor("epoch")],  # Log learning rate every epoch
                             progress_bar_refresh_rate=1)
        trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
        trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
        return trainer

    def get_pretrained_filename(self, save_name=None):
        if save_name is None:
            save_name = self.get_model_name()
        pretrained_filename = os.path.join(self.checkpoints_dir, save_name + ".ckpt")
        return pretrained_filename

    def get_model_name(self):
        model_type = type(self.model)
        model_name = model_type.__name__
        return model_name
