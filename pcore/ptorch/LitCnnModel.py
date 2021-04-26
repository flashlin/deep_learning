import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim.adam import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# def one_hot(y, classes_num):
#     y = y.view(-1, 1)
#     y_onehot = torch.FloatTensor(1, classes_num)
#     y_onehot.zero_()
#     y_onehot.scatter_(1, y, 1)
#     return y_onehot


from py.Info import info


class CnnModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            # nn.Linear(4 * 7 * 7, 10),
            nn.Linear(36, 10),
            # nn.ReLU(inplace=True),
        )

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = self.cnn_layers(x)
        x = x.view(batch_size, -1)
        x = self.linear_layers(x)
        return x


class LitModel(pl.LightningModule):
    def __init__(self, nn_model):
        super().__init__()
        self.model = nn_model

    def forward(self, x):
        # batch_size, channels, width, height = x.size()
        x = self.model.forward(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.model.forward(x)
        loss = self.model.loss_func(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model.forward(x)
        loss = self.model.loss_func(logits, y)
        self.log('val_loss', loss)

        val_acc = self.model.accuracy_score(logits, y)
        self.log("val_acc", val_acc)

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return optimizer

    # def train_dataloader(self):
    #     dataset = self.dataset
    #     dataset.batch_size = self.batch_size
    #     train_loader, val_loader = dataset.get_train_validation_loaders()
    #     return train_loader


class LitCnnModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
        )

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = self.cnn_layers(x)
        x = x.view(batch_size, -1)
        x = self.linear_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = self.loss_func(logits, y)
        assert loss >= 0
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_func(logits, y)
        assert loss >= 0
        self.log('val_loss', loss)

        # acc
        _, y_hat = torch.max(logits, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), y.cpu())
        val_acc = torch.tensor(val_acc)
        self.log("val_acc", val_acc)

    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return optimizer

    def save_pretrain_data(self, save_dir):
        self.dataset.save_classes_names(f"{save_dir}/LitCnnModel.names")


    # def train_dataloader(self):
    #     dataset = self.dataset
    #     dataset.batch_size = self.batch_size
    #     train_loader, val_loader = dataset.get_train_validation_loaders()
    #     return train_loader
