import torch
import torch.utils.data
from PIL import Image
import pandas as pd
import os
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

class CSVImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, names=None, header=None, sep=",", root=None, transforms=None):
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = T.Compose([T.ToTensor()])
        self.data = pd.read_csv(csv_path, sep=sep, header=header, names=names)
        self.root = root
        self.classes = list(self.data[1].unique())
        self.num_classes = len(self.classes)
  
    def __getitem__(self, idx):
        images = self.data[0]
        labels = self.data[1]

        img_path =  os.path.join(self.root, images[idx])
        label = torch.as_tensor(labels[idx], dtype=torch.int64)
        img = Image.open(img_path).convert("RGB")

        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    csv_f = '/scratch/ccekwk/flickr_style/test_new.txt'
    root="/scratch/ccekwk/flickr_style/images"
    ds = CSVImageDataset(csv_f, names=None, header=None, sep= " ", root = root)
    print(ds.num_classes)
    print(ds.__len__())
    print(ds.data)

