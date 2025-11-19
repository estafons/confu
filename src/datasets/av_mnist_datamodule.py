import sys
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import random
# Add the path to your AVMNIST dataset library
avmnist_path = Path(__file__).resolve().parent.parent.parent / "libs"
sys.path.append(str(avmnist_path))
print(f"Added {avmnist_path} to sys.path")

from multibench.datasets.avmnist.get_data import get_dataloader  # Adjust import as needed

# Mapping from label int to text
label_to_text = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

# Custom Dataset returning image, audio, and text

def get_text_from_label(label):

    templates = [
        "an image of a {}",
        "a photo of a {}",
        "a picture of a {}",
        "this is a {}",
        "there is a {}",
        "it is a {}",
        "the number is a {}",
        "the digit is a {}",
        "a drawing of a {}",
        "a sketch of a {}",
    ]

    # random choice
    
    template = random.choice(templates)

    return template.format(label_to_text[label])


class AVMNISTDataset(Dataset):
    def __init__(self, dataloader):
        self.data = dataloader.dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, audio, label = self.data[idx]  # Assuming get_dataloader returns tuples like this

        # Convert from numpy.ndarray to torch.Tensor
        image = torch.tensor(image, dtype=torch.float32)
        audio = torch.tensor(audio, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        # Create text representation
        text = f"an image of a {label_to_text[int(label)]}"

        return image, audio, text, label


# Lightning DataModule
class AVMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_loader_raw, val_loader_raw, test_loader_raw = get_dataloader(
            self.data_path,
            batch_size=self.batch_size,
            train_shuffle=True
        )
        self.train_dataset = AVMNISTDataset(train_loader_raw)
        self.val_dataset = AVMNISTDataset(val_loader_raw)
        self.test_dataset = AVMNISTDataset(test_loader_raw)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


