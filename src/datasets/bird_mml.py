import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchaudio
import pandas as pd
from pathlib import Path
import random
import pytorch_lightning as pl
from src.datasets.bird_triplet_datamodule import BirdTripletDataset

class BirdCaptionTrainDataset(Dataset):
    """Dataset for image–audio–caption data per species."""
    def __init__(self, paired_csv, taxa_csv, audio_dir, transform=None, audio_transform=None):
        self.df = pd.read_csv(paired_csv)
        self.taxa_df = pd.read_csv(taxa_csv)
        self.audio_dir = Path(audio_dir)

        # Map scientific -> common name
        self.scientific_to_common = dict(zip(self.taxa_df['scientific_name'], self.taxa_df['common_name']))

        # Label encoding
        self.scientific_names = sorted(self.df['scientific_name'].unique())
        self.label2name = {i: self.scientific_to_common.get(name, name) for i, name in enumerate(self.scientific_names)}
        self.df['label'] = self.df['scientific_name'].apply(lambda x: self.scientific_names.index(x))

        # Group by label
        self.audio_by_label = self.df.groupby('label')['audio_file'].apply(list).to_dict()
        self.image_by_label = self.df.groupby('label')['photo_file'].apply(list).to_dict()
        self.caption_by_label = self.df.groupby('label')['combined_caption'].apply(list).to_dict()

        self.labels = sorted(list(self.audio_by_label.keys()))

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.audio_transform = audio_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Pick a random species each time
        label = random.choice(self.labels)

        audio_file = random.choice(self.audio_by_label[label])
        image_file = random.choice(self.image_by_label[label])
        caption = random.choice(self.caption_by_label[label])
        # print(f"Selected label: {label}, audio: {audio_file}, image: {image_file}, caption: {caption}")
        # --- Load image ---
        try:
            image = Image.open(image_file).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"⚠️ Failed to load image: {image_file} ({e})")
            image = torch.zeros(3, 224, 224)
            print(label)

        # --- Load audio ---
        try:
            waveform, sr = torchaudio.load(audio_file)
            if self.audio_transform:
                waveform = self.audio_transform(waveform)
        except Exception as e:
            # print(f"⚠️ Failed to load audio: {audio_file} ({e})")
            # print(f'class is {self.label2name[label]}')
            waveform = torch.zeros(1, 220500)  # 10 sec @ 22.05 kHz fallback
            if self.audio_transform:
                waveform = self.audio_transform(waveform)

        # --- Caption ---
        caption = caption or f"An image of a {self.label2name[label]}"

        return image, waveform, caption, label


# ------------------- Lightning DataModule -------------------
class BirdCaptionDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, test_csv, taxa_csv, audio_dir, image_sources,
                 batch_size=4, num_workers=4, transform=None, audio_transform=None):
        super().__init__()
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.taxa_csv = taxa_csv
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.audio_transform = audio_transform
        self.image_sources = image_sources

    def setup(self, stage=None):
        self.train_dataset = BirdCaptionTrainDataset(
            paired_csv=self.train_csv,
            taxa_csv=self.taxa_csv,
            audio_dir=self.audio_dir,
            transform=self.transform,
            audio_transform=self.audio_transform
        )

        # Use your existing BirdTripletDataset for test set
        
        self.test_dataset = BirdTripletDataset(  # reuse your previous test dataset
            audio_csv=self.test_csv,
            image_sources=self.image_sources,
            species_csv=self.taxa_csv,
            audio_dir=self.audio_dir,
            split="test",
            transform=self.transform,
            audio_transform=self.audio_transform
        )

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
