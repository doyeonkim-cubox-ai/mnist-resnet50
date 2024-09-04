from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
import os


class Mydata(Dataset):
    def __init__(self, data_dir: str,
                 transform=None):
        self.data_dir = data_dir
        self.img_paths = glob(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        self.img_path = os.path.join(self.img_paths[idx])
        self.img = Image.open(self.img_path)
        self.x = self.transform(self.img)
        self.label = os.path.split(self.img_path)[0][-1]
        return self.x, self.label
