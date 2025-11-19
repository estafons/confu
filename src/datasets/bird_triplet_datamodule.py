from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchaudio
import pandas as pd
import random
from pathlib import Path
from PIL import Image
import pytorch_lightning as pl

class BirdTripletDataset(Dataset):
    def __init__(self, audio_csv, image_sources, species_csv, audio_dir, split='test', transform=None, audio_transform=None):
        """
        Args:
            audio_csv (str): path to audio metadata csv
            image_sources (list[tuple[str, str]]): list of (image_csv_path, image_dir_path)
            species_csv (str): path to species metadata csv
            audio_dir (str): path to folder with audio files
            split (str): data split ("train", "test", etc.)
        """
        print("Loading dataset...")
        print(f"  Audio CSV: {audio_csv}")
        self.audio_df = pd.read_csv(audio_csv)
        self.species_df = pd.read_csv(species_csv)

        # combine all image csvs and remember their base directory
        image_dfs = []
        for csv_path, img_dir in image_sources:
            df = pd.read_csv(csv_path)
            df["img_dir"] = img_dir
            image_dfs.append(df)
        self.image_df = pd.concat(image_dfs, ignore_index=True)

        # Filter for split
        self.audio_df = self.audio_df[self.audio_df['split'] == split]
        self.image_df = self.image_df[self.image_df['split'] == split]
        
        # Create label → species map
        self.label2name = dict(zip(self.species_df['label'], self.species_df['common_name']))
        
        # Group by label for random pairing
        self.image_by_label = self.image_df.groupby('label').apply(
            lambda g: list(zip(g['asset_id'], g['img_dir']))
        ).to_dict()
        self.audio_by_label = self.audio_df.groupby('label')['asset_id'].apply(list).to_dict()

        self.labels = sorted(list(set(self.audio_by_label.keys()) & set(self.image_by_label.keys())))
        
        self.audio_dir = Path(audio_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.audio_transform = audio_transform

    def __len__(self):
        return sum(len(self.audio_by_label[lbl]) for lbl in self.labels)

    def __getitem__(self, idx):
        # pick a random label
        label = random.choice(self.labels)
        audio_id = random.choice(self.audio_by_label[label])
        image_id, img_dir = random.choice(self.image_by_label[label])
        
        # --- Load image ---
        try:
            img_path = Path(img_dir) / f"{image_id}.jpg"
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            
            # --- Load audio ---
            audio_path = self.audio_dir / f"{audio_id}.wav"
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading data for label {label}, audio_id {audio_id}, image_id {image_id}: {e}")
            # Return a random sample instead
            return self.__getitem__(random.randint(0, len(self) - 1))
        if self.audio_transform:
            waveform = self.audio_transform(waveform)
        
        # --- Text prompt ---
        name = self.label2name[label]
        text = f"an image of a {name}"
        
        return image, waveform, text, label

class BirdTripletDataModule(pl.LightningDataModule):
    def __init__(self, audio_csv, image_sources, species_csv, audio_dir,
                 batch_size=4, num_workers=4, split_map=None,
                 transform=None, audio_transform=None):
        """
        image_sources: list of (image_csv_path, image_dir_path)
        split_map: dict mapping split names to use, e.g. {"train": "train", "val": "val", "test": "test"}
        """
        super().__init__()
        self.audio_csv = audio_csv
        self.image_sources = image_sources
        self.species_csv = species_csv
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.audio_transform = audio_transform
        self.split_map = split_map or {"train": "train", "val": "val", "test": "test"}

        self.datasets = {}

    def setup(self, stage=None):
        # Create datasets for each split
        for key, split_name in self.split_map.items():
            self.datasets[key] = BirdTripletDataset(
                audio_csv=self.audio_csv,
                image_sources=self.image_sources,
                species_csv=self.species_csv,
                audio_dir=self.audio_dir,
                split=split_name,
                transform=self.transform,
                audio_transform=self.audio_transform
            )
        
        print("Dataset sizes:")
        for key, dataset in self.datasets.items():
            print(f"  {key}: {len(dataset)}")

    def train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.datasets["val"], batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)
