import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# Resolve the absolute path to ../../libs/multibench relative to this file
multibench_path = Path(__file__).resolve().parent.parent.parent / "libs"
sys.path.append(str(multibench_path))
print(f"Added {multibench_path} to sys.path")


from multibench.datasets.affect.get_data import get_dataloader



# Augmentation functions you provided
def identity(x):
    return x

def permute(x):
    idx = torch.randperm(x.shape[0])
    return x[idx]

def noise(x):
    noise = torch.randn(x.shape) * 0.1
    return x + noise.to(x.device)

def drop(x):
    drop_num = x.shape[0] // 5
    x_aug = torch.clone(x)
    drop_idxs = np.random.choice(x.shape[0], drop_num, replace=False)
    x_aug[drop_idxs] = 0.0
    return x_aug

def augment_single(x_batch):
    v1 = x_batch
    v2 = torch.clone(v1)
    transforms = [permute, noise, drop, identity]

    for i in range(x_batch.shape[0]):
        t_idxs = np.random.choice(4, 1, replace=False)
        t = transforms[t_idxs[0]]
        v2[i] = t(v2[i])

    return v2


# Custom Dataset returning modality augmented pairs and label
class AffectAugmentedDataset(Dataset):
    def __init__(self, dataloader, samples_order, stage=None, dataset_name=None):
        self.data = dataloader.dataset
        self.stage = stage
        self.dataset_name = dataset_name
        self.samples_order = samples_order

    def __len__(self):
        return len(self.data)

    def _get_sample_orders(self, sample):
        if self.stage == 'train':
            #return [augment_single(sample[i]) for i in self.samples_order]
            return [sample[i] for i in self.samples_order]
        elif self.stage in ['val', 'test']:
            return [sample[i] for i in self.samples_order]
        else:
            raise ValueError(f"Unsupported stage: {self.stage}. Supported stages are 'train', 'val', 'test'.")

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.dataset_name in ["mosi", "mosei"]:
            label = 0 if sample[3] < 0 else 1
        elif self.dataset_name in ["humor", "sarcasm"]:
           # print(f'types of sample 3: {sample[3].dtype} 2: {sample[2].dtype} 1: {sample[1].dtype} 0: {sample[0].dtype}')
            if sample[3] == -1.0:
                label = torch.tensor(0, dtype=torch.long)  # <- long tensor with int value
            else: 
                label = sample[3].detach().clone().squeeze(-1).long() # <- float tensor with int value

        # round label to closest integer. return as integer
    #    label = int(torch.round(sample[3]).item())  # Ensure label is an integer



        # # Create two augmentations per modality using augment_single
        
        
        if len(self.samples_order) == 2:
            modality1, modality2 = self._get_sample_orders(sample)
            return modality1, modality2, label
        elif len(self.samples_order) == 3:
            modality1, modality2, modality3 = self._get_sample_orders(sample)
            # if self.stage == 'train':
            #     modality1 = augment_single(sample[0])
            #     modality2 = augment_single(sample[1])
            #     modality3 = augment_single(sample[2])
           # else:
            # For validation/test, return the original modalities without augmentation
                
            return modality1, modality2, modality3, label  # Return the three modalities and the label
        else:
            raise ValueError(f"Unsupported number of modalities: {len(self.samples_order)}. Supported orders are 2 or 3.")


# Lightning DataModule
class AffectDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=16, samples_order=[0,1,2], 
                 pickle_name=None, dataset_name=None, ):
        super().__init__()
        self.data_path = Path(__file__).resolve().parent.parent.parent / "data" / "multibench" / pickle_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.samples_order = samples_order

    def setup(self, stage=None):
        train_loader_raw, val_loader_raw, test_loader_raw = get_dataloader(
            self.data_path,
            robust_test=False,
            batch_size=self.batch_size,
            train_shuffle=True, max_pad=True, max_seq_len=50
        )
        self.train_dataset = AffectAugmentedDataset(train_loader_raw, stage="train", dataset_name=self.dataset_name, samples_order=self.samples_order)
        self.val_dataset = AffectAugmentedDataset(val_loader_raw, stage="val", dataset_name=self.dataset_name, samples_order=self.samples_order)
        self.test_dataset = AffectAugmentedDataset(test_loader_raw, stage="test", dataset_name=self.dataset_name, samples_order=self.samples_order)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)


if __name__ == "__main__":
    # Example usage
    pickle_name = 'mosi_data.pkl'
    mosi_dm = AffectDataModule(batch_size=16, num_workers=4, pickle_name=pickle_name, dataset_name='mosi')
    mosi_dm.setup()

    train_loader = mosi_dm.train_dataloader()
    for batch in train_loader:
        
        mod1, mod2, mod3, label = batch
        print(f"Modality 1 shape: {mod1.shape}, Modality 2 shape: {mod2.shape}, Modality 3 shape: {mod3.shape}, Label: {label}")
        break