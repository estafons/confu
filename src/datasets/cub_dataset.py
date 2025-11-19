import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


class CUBBirdsDataset(Dataset):
    def __init__(self, root_dir, transform=None, dummy_audio=None, dummy_text="dummy text"):
        """
        Args:
            root_dir (str): Path to the root image directory (one subfolder per class).
            transform (callable, optional): Transform to apply on images.
            dummy_audio (torch.Tensor): Preloaded dummy audio tensor to return for every sample.
            dummy_text (str): Dummy text to return for every sample.
        """
        self.name = "cub_birds"
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.dummy_audio = dummy_audio
        self.dummy_text = dummy_text

        # Get class folders and label mapping
        self.classes = sorted([
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Collect (image_path, label)
        self.samples = []
        for cls_name in self.classes:
            cls_path = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_path):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, self.dummy_audio, self.dummy_text, label

    def get_texts(self):
        return self.classes

