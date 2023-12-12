from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from pathlib import Path
from torchvision.io import read_image


class TestDataset(Dataset):
    def __init__(self, directory):
        """
        A simple torch dataset for handling testing data only.
        Given a directory, allow for easy access to images contained within that directory.

        Handles all required transforms for the scene images.

        :param directory: The directory path.
        """
        self.filenames = [(Path(directory).resolve() / fn) for fn in os.listdir(directory)]

    def __getitem__(self, item):
        image = read_image(str(self.filenames[item]))
        resize = transforms.Compose([transforms.Resize((224, 224), antialias=None)])
        image = resize(image)
        image = image.float()
        image = image / 255.0

        return image, self.filenames[item].stem

    def __len__(self):
        return len(self.filenames)
