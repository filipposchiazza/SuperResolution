import os
import torch
import torch.nn.functional as F
from torchvision.io import read_image
import torch.utils.data as data



class SuperResDataset(data.Dataset):
    def __init__(self, img_dir, transform=None, factor=4):
        self.img_dir = img_dir
        self.transform = transform
        self.img_filenames = os.listdir(img_dir)
        self.factor = factor

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_name = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        image = image[:3, :, :]
        image = image / 255.0
        lowres_size = (image.shape[1] // self.factor, image.shape[2] // self.factor)
        image_lowres = F.interpolate(image.unsqueeze(0), size=lowres_size, mode='bicubic')
        image_lowres = torch.clamp(image_lowres, 0.0, 1.0)
        return image_lowres, image
    


def load_superres_data(folder,
                       batch_size,
                       validation_split,
                       seed=123):
    dataset = SuperResDataset(folder, transform=None)
    val_len = int(validation_split * len(dataset))
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = data.random_split(dataset,
                                                   lengths=[train_len, val_len],
                                                   generator=generator)
    train_dataloader = data.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
    val_dataloader = data.DataLoader(val_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)
    return train_dataset, val_dataset, train_dataloader, val_dataloader
    