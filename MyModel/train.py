import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

from ptorch.ImageClassificationDataset import ImageClassificationDataset, ImageClassification
from ptorch.LitCnnModel import LitCnnModel
from ptorch.LitTrainer import LitTrainer

cnn_filters = [
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ],
    [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ],
    [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ],
    [
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
    ],
    [
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ],
    [
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 0],
    ],
    [
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 0],
    ],
    [
        [0, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
    ],
]


def apply_image_filter(image, filter_index=0):
    global cnn_filters
    convolved = convolve2d(image, cnn_filters[filter_index])
    k1 = np.array([
        [1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    convolved = convolve2d(convolved, k1)
    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    plt.title("Original image")
    plt.axis("off")
    plt.imshow(image, cmap="gray")

    plt.subplot(122)
    plt.title("Convolved image")
    plt.axis("off")
    plt.imshow(convolved, cmap="gray")

    plt.show()
    return convolved


def conv2d_3x3(filter_index=0):
    global cnn_filters

    conv = nn.Conv2d(in_channels=1,
                     out_channels=1,
                     kernel_size=3,
                     stride=1,
                     bias=None,
                     padding=0
                     )
    kernel_3x3 = torch.tensor([[cnn_filters[filter_index]]], dtype=torch.float16)
    # plt.imshow(kernel_3x3)
    # plt.show()
    print(f"{conv.weight.data}")
    print(f"k = {kernel_3x3}")
    conv.weight.data = kernel_3x3
    # conv.weight.requires_grad = False
    return conv


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 7 * 7, 10)
        )
        # self.embedding = nn.Embedding(input_size, hidden_size)
        # self.gru = nn.GRU(hidden_size, hidden_size)
        # self.fc1 = nn.Linear(64 * 7 * 7, 1024)

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # plt.figure(figsize=(8,8))
        # plt.subplot(111)
        # plt.title("Original image")
        # plt.axis("off")
        # plt.imshow(x)
        # plt.show()

        x = self.cnn_layers(x)
        x = x.view(batch_size, -1)
        x = self.linear_layers(x)
        return x

    # def loss_func(self):
    #     return torch.nn.CrossEntropyLoss()


def main():
    images_dir = f"D:/Demo/archive/trainingSet/trainingSet"
    dataset = ImageClassificationDataset(images_dir)

    # trainer = LitModelTrainer(MyModel())
    # trainer.fit_model(dataset)

    # trainer.load_pretrained_model("MyModel")
    # pred = trainer.predict_image("D:/Demo/archive/testSample/img_1.jpg")
    # info(f"{pred}")

    # image_path = f"D:/Demo/training/trainingImages/images/IMG_2116.jpg"
    # image = np.array(Image.open(image_path).convert('L'))
    # apply_image_filter(image, 0)

    #run_train(dataset)
    test_model(dataset)


def test_model(dataset):
    trainer = LitTrainer()
    model = trainer.load_pretrained_model(LitCnnModel())
    model.freeze()
    # train_data, valid_data = dataset.get_train_validation_loaders()
    print(dataset.classes_dict.names)
    for n in range(1, 40):
        image_path = f"D:/Demo/archive/testSample/img_{n}.jpg"
        image_classification = ImageClassification()
        img = image_classification.get_image_data(image_path)
        output = model(img)
        # print(f"output = {output}")
        _, preds_tensor = torch.max(output, 1)
        # print(f"{torch.max(output, 1)}")
        preds = np.squeeze(preds_tensor.numpy())
        print(f"img_{n} = {preds}")


def run_train(dataset):
    trainer = LitTrainer()
    trainer.batch_size = 64
    trainer.max_epochs = 100
    trainer.fit_model(LitCnnModel(), dataset)


if __name__ == '__main__':
    main()
