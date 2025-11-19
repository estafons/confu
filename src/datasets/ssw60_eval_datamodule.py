import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
import warnings
import torch.nn as nn
import torchaudio


warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")



class BirdVideoDataset(Dataset):
    """
    Dataset that returns (image, audio, text, label) for each sample.

    Parameters
    ----------
    paired_csv : str | Path
        CSV file containing asset_id, label, and other metadata.
    taxa_csv : str | Path
        CSV with label ↔ common_name mapping.
    video_dir : str | Path
        Folder containing .mp4 video files named <asset_id>.mp4
    transform : torchvision transform, optional
        Image transform (default: resize and normalize to 224x224)
    audio_transform : callable, optional
        Optional audio transform.
    frame_sample_second : float
        Second to sample the frame from (default: 2.0)
    """

    def __init__(
        self,
        paired_csv,
        taxa_csv,
        video_dir,
        transform=None,
        audio_transform=None,
        frame_sample_second=2.0,
    ):
        self.name = "ssw60_bird_videos"
        self.df = pd.read_csv(paired_csv)
        self.taxa_df = pd.read_csv(taxa_csv)
        self.video_dir = Path(video_dir)

        # Build label → common name mapping
        self.label2name = dict(
            zip(self.taxa_df["label"], self.taxa_df["common_name"])
        )

        # Default image transform
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        self.audio_transform = audio_transform
        self.frame_sample_second = frame_sample_second

    def __len__(self):
        return len(self.df)


    def _extract_audio(self, video_path):
        """Extract audio waveform from a video using torchaudio."""
        waveform, sr = torchaudio.load(str(video_path))
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)(waveform)
        if self.audio_transform:
            
            waveform = self.audio_transform(waveform)
            
        return waveform, sr
    
    def _extract_frames(self, video_path, n_frames=8):
        """
        Extract `n_frames` evenly spaced frames from the video.
        Returns a list of PIL Images.
        """
        frames = []
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Compute the middle frame of each segment
            for i in range(n_frames):
                start = i * total_frames // n_frames
                end = (i + 1) * total_frames // n_frames
                middle_frame = (start + end) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                success, frame = cap.read()
                if success and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
                else:
                    frames.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))
            cap.release()
        except Exception:
            # fallback: list of black images
            frames = [Image.new("RGB", (224, 224), color=(0, 0, 0)) for _ in range(n_frames)]
        return frames

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        asset_id = row["asset_id"]
        label = int(row["label"])

        video_path = self.video_dir / f"{asset_id}.mp4"
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # --- Images: 8 frames ---
        frames = self._extract_frames(video_path, n_frames=8)
        frames = [self.transform(f) for f in frames]  # apply transform to each frame

        # --- Audio ---
        waveform, sr = self._extract_audio(video_path)

        # --- Text (caption) ---
        species_name = self.label2name.get(label)
        caption = f"an image of a {species_name}"

        return frames, waveform, caption, label
    
    def get_texts(self):
        """
        Return captions in order of label_id (0..N-1) for zero-shot evaluation.
        """
        id2name = {row['label']: row['common_name'] for _, row in self.taxa_df.iterrows()}
        return [f"{id2name[i]}" for i in range(len(id2name))]

class ToMono(nn.Module):
    def forward(self, waveform):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

class BirdVideoDatasetSingleFrame(Dataset):
    """
    Dataset that returns (image, audio, text, label) for each sample.
    
    Parameters
    ----------
    paired_csv : str | Path
        CSV file containing asset_id, label, and other metadata.
    taxa_csv : str | Path
        CSV with label ↔ common_name mapping.
    video_dir : str | Path
        Folder containing .mp4 video files named <asset_id>.mp4
    transform : torchvision transform, optional
        Image transform (default: resize and normalize to 224x224)
    audio_transform : callable, optional
        Optional audio transform.
    frame_sample_second : float
        Second to sample the frame from (default: 2.0)
    """
    
    def __init__(
        self,
        paired_csv,
        taxa_csv,
        video_dir,
        transform=None,
        audio_transform=None,
        frame_sample_second=2.0,
    ):
        self.name = "ssw60_bird_videos"
        self.df = pd.read_csv(paired_csv)
        self.taxa_df = pd.read_csv(taxa_csv)
        self.video_dir = Path(video_dir)
        
        # Build label → common name mapping
        self.label2name = dict(
            zip(self.taxa_df["label"], self.taxa_df["common_name"])
        )
        
        # Default image transform
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.audio_transform = audio_transform
        self.frame_sample_second = frame_sample_second
        
    def __len__(self):
        return len(self.df)
    
    def _extract_audio(self, video_path):
        """Extract audio waveform from a video using torchaudio."""
        try:
            waveform, sr = torchaudio.load(str(video_path))
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=22050)(waveform)
            if self.audio_transform:
                waveform = self.audio_transform(waveform)
                

            return waveform, sr
        except Exception:
            # fallback: silent waveform (10s at 22.05kHz)
            return torch.zeros(1, 220500), 22050
    
    def _extract_frame_at_second(self, video_path, target_second):
        """
        Extract a single frame at the specified second from the video.
        
        Parameters
        ----------
        video_path : Path
            Path to the video file
        target_second : float
            Second to extract the frame from
            
        Returns
        -------
        PIL.Image
            Extracted frame as PIL Image, or black image if extraction fails
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Calculate frame position
            target_frame = int(target_second * fps)
            
            # Clamp to valid range
            target_frame = min(max(0, target_frame), total_frames - 1)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            success, frame = cap.read()
            cap.release()
            
            if success and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame)
            else:
                return Image.new("RGB", (224, 224), color=(0, 0, 0))
                
        except Exception:
            # fallback: black image
            return Image.new("RGB", (224, 224), color=(0, 0, 0))
    
    def _extract_frames(self, video_path, n_frames=8):
        """
        Extract n_frames evenly spaced frames from the video.
        Returns a list of PIL Images.
        """
        frames = []
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Compute the middle frame of each segment
            for i in range(n_frames):
                start = i * total_frames // n_frames
                end = (i + 1) * total_frames // n_frames
                middle_frame = (start + end) // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
                success, frame = cap.read()
                if success and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
                else:
                    frames.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))
            cap.release()
        except Exception:
            # fallback: list of black images
            frames = [Image.new("RGB", (224, 224), color=(0, 0, 0)) for _ in range(n_frames)]
        return frames
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        asset_id = row["asset_id"]
        label = int(row["label"])
        video_path = self.video_dir / f"{asset_id}.mp4"
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # --- Single frame at specified second ---
        frame = self._extract_frame_at_second(video_path, self.frame_sample_second)
        frame_tensor = self.transform(frame)  # apply transform to single frame
        
        # --- Audio ---
        waveform, sr = self._extract_audio(video_path)
        
        # --- Text (caption) ---
        species_name = self.label2name.get(label)
        caption = f"an image of a {species_name}"
        
        return frame_tensor, waveform, caption, label
    
    def get_texts(self):
        """
        Return captions in order of label_id (0..N-1) for zero-shot evaluation.
        """
        id2name = {row['label']: row['common_name'] for _, row in self.taxa_df.iterrows()}
        return [f"{id2name[i]}" for i in range(len(id2name))]