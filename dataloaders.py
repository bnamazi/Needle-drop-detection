import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, io
import PIL
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

import os
import math

transform_train = torch.nn.Sequential(
    transforms.Resize((300, 300), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomApply(
        torch.nn.ModuleList([transforms.RandAugment()]),
        0.85,
    ),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.4270, 0.2752, 0.2710), (0.2403, 0.2212, 0.2203)),
)

transform_val = torch.nn.Sequential(
    transforms.Resize((300, 300), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize((0.4270, 0.2752, 0.2710), (0.2403, 0.2212, 0.2203)),
)


class NeedleDropImageDataset(Dataset):
    def __init__(
        self, data_dir, transform=None, train=True, split="train", left_right=False
    ):

        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        self.split = split
        self.label_path = os.path.join(f"{self.split}.csv")
        self.df = pd.read_csv(self.label_path)
        if self.train:
            self.df = shuffle(self.df)
        self.left_right = left_right

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        line = self.df.iloc[[idx]]

        case_id = int(line["case_id"])
        frame_num = int(line["frame_id"])

        img_path = os.path.join(
            self.data_dir,
            f"caseid_{str(case_id).zfill(6)}_fps1",
            f"caseid_{str(case_id).zfill(6)}_fps1Frame{str(frame_num).zfill(3)}.jpg",
        )

        image = torchvision.io.read_image(img_path)
        if self.left_right:
            label = torch.tensor([int(line["left"]), int(line["right"])])
        else:
            label = int(line["dropped"])

        return self.transform(image), label, img_path


if __name__ == "__main__":
    from data_augmentation import transform_train, transform_val
    from torch.utils.data import Dataset, DataLoader
    from torchvision.transforms import Compose, Lambda, Resize

    transform1 = torch.nn.Sequential(
        transforms.Resize(
            (300, 300), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        # transforms.RandomApply(torch.nn.ModuleList([transforms.RandAugment()]), 0.85),
        transforms.ConvertImageDtype(torch.float),
    )

    dataset = NeedleDropImageDataset(
        data_dir=".\data\images",
        transform=transform1,
        train=False,
        split="train",
        left_right=True,
    )

    data = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=1,
        pin_memory=False,
    )
    iterator = iter(data)

    # for _ in range(10):

    inputs, label, _ = iterator.next()

    # out = transform_val(inputs)

    print(inputs.shape, label)

    grid = torchvision.utils.make_grid(torch.squeeze(inputs), nrow=5)

    img = torchvision.transforms.ToPILImage()(grid)
    img.show()
    # label.cpu().detach().numpy()
    # print(inputs.max())
