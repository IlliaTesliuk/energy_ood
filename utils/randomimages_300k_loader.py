import numpy as np
import torch

class RandomImages(torch.utils.data.Dataset):

    def __init__(self, bin_file, size, transform=None):

        # '../data/80million/tiny_images.bin'
        data_file = open(bin_file, "rb")

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            return np.fromstring(data, dtype='uint8').reshape(32, 32, 3)#, order="F")

        self.load_image = load_image
        self.offset = 0     # offset index
        self.size = size

        self.transform = transform
        
    def __getitem__(self, index):
        index = (index + self.offset) % self.size

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return self.size
