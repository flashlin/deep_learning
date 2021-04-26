import glob
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
from pytorch_lightning import Trainer, seed_everything
from ptorch.ImageClassificationDataset import ImageClassificationDataset
from ptorch.LitCnnModel import LitCnnModel
from py.Info import info
from py.io import get_folder_list


def get_model_name(model):
    model_type = type(model)
    model_name = model_type.__name__
    return model_name


class LitTrainer:
    checkpoints_dir = "d:/demo/training/checkpoints"
    max_epochs = 10
    # seed_num = 42
    batch_size = 33

    def fit_model(self, model, dataset):
        model = self.load_pretrained_model(model)
        model.dataset = dataset
        model.save_pretrain_data(f"{self.checkpoints_dir}")

        # seed_everything(self.seed_num)
        trainer = self.create_trainer(model)

        dataset.batch_size = self.batch_size
        train_dataloader, val_dataloader = dataset.get_train_validation_loaders()
        trainer.fit(model, train_dataloader, val_dataloader)

    def create_trainer(self, model):
        cuda_available = torch.cuda.is_available
        model_checkpoint = ModelCheckpoint(save_weights_only=True,
                                           mode="min",
                                           monitor="val_acc",
                                           save_last=True)
        save_name = get_model_name(model)
        trainer = pl.Trainer(default_root_dir=os.path.join(self.checkpoints_dir, save_name),
                             checkpoint_callback=model_checkpoint,
                             gpus=1 if cuda_available is True else 0,
                             max_epochs=self.max_epochs,
                             deterministic=True,
                             #auto_scale_batch_size='binsearch',
                             #callbacks=[LearningRateMonitor("epoch")],
                             #progress_bar_refresh_rate=1,
                             )
        #trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
        trainer.logger._default_hp_metric = None

        #trainer.tune(model)
        return trainer

    def load_pretrained_model(self, model):
        model_name = get_model_name(model)
        pretrained_filename = self.get_last_checkpoints_filename(model_name)
        if pretrained_filename is not None and os.path.isfile(pretrained_filename):
            info(f"Found pretrained model at {pretrained_filename}, loading...")
            model = type(model).load_from_checkpoint(pretrained_filename)
        return model

    def get_last_checkpoints_filename(self, model_name):
        version_dirs = sorted(get_folder_list(f"{self.checkpoints_dir}/{model_name}/lightning_logs"), reverse=True)
        if len(version_dirs) == 0:
            return None
        ckpt = f"{version_dirs[0]}/checkpoints/last.ckpt"
        if not os.path.isfile(ckpt):
            return None
        return ckpt
