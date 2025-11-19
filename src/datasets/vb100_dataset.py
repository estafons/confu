import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
import librosa
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
import warnings
import librosa

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")



class VB100Dataset(Dataset):
    """
    Dataset that returns (frames[8], audio, caption, label_id)
    where frames is a list of 8 images uniformly sampled across the video.
    """

    def __init__(
        self,
        txt_file1,
        video_root,
        txt_file2=None,
        transform=None,
        audio_transform=None,
        n_frames=8,
    ):
        self.name = "vb100"
        self.video_root = Path(video_root)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.audio_transform = audio_transform
        self.n_frames = n_frames

        # Parse text files
        data = []
        raw_labels = set()
        for txt_file in filter(None, [txt_file1, txt_file2]):
            with open(txt_file, "r") as f:
                for line in f:
                    path, label = line.strip().split()
                    species = Path(path).parent.name.replace("_", " ")
                    raw_label = int(label)
                    raw_labels.add(raw_label)
                    data.append({
                        "path": path,
                        "species": species,
                        "raw_label": raw_label
                    })

        # Map raw labels -> contiguous 0..N-1
        self.raw_to_id = {raw: i for i, raw in enumerate(sorted(raw_labels))}
        self.id2name = {}
        for item in data:
            item["label_id"] = self.raw_to_id[item["raw_label"]]
            self.id2name[item["label_id"]] = item["species"]

        self.df = pd.DataFrame(data)

    def __len__(self):
        return len(self.df)

    def _extract_frames_uniform(self, video_path, n_frames=8):
        """Extract `n_frames` uniformly spaced frames from the video."""
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [Image.new("RGB", (224, 224), color=(0, 0, 0)) for _ in range(n_frames)]

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return [Image.new("RGB", (224, 224), color=(0, 0, 0)) for _ in range(n_frames)]

        # Compute equally spaced frame indices (middle of each segment)
        segment_length = total_frames / n_frames
        frame_indices = [int(segment_length * (i + 0.5)) for i in range(n_frames)]

        for fid in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            success, frame = cap.read()
            if success and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            else:
                frames.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))
        cap.release()
        return frames

    def _extract_audio(self, video_path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
            warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
            try:
                waveform, sr = librosa.load(str(video_path), sr=22050, mono=True)
                if waveform is None or len(waveform) == 0:
                    raise ValueError("Empty waveform returned")
            except Exception:
                sr = 22050
                waveform = torch.zeros(1, 220500)
                if self.audio_transform:
                    waveform = self.audio_transform(waveform)
                return waveform, sr
            waveform = torch.tensor(waveform).unsqueeze(0)
            if self.audio_transform:
                waveform = self.audio_transform(waveform)
            return waveform, sr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = self.video_root / row["path"]
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # --- Extract 8 frames ---
        frames = self._extract_frames_uniform(video_path, n_frames=self.n_frames)
        frames = [self.transform(f) for f in frames]  # list of 8 tensors

        # --- Extract audio ---
        waveform, sr = self._extract_audio(video_path)

        # --- Caption ---
        caption = f"an image of a {row['species']}"

        return frames, waveform, caption, row["label_id"]

    def get_texts(self):
        """Return captions in order of label_id (0..N-1) for zero-shot evaluation."""
        return [f"{self.id2name[i]}" for i in range(len(self.id2name))]

    def get_labels(self):
        """Return contiguous label IDs in order 0..N-1"""
        return list(range(len(self.id2name)))

class VB100DatasetSingleFrame(Dataset):
    """
    Dataset that returns (frame, audio, caption, label_id)
    where frame is a single image sampled at the specified second.
    """

    def __init__(
        self,
        txt_file1,
        video_root,
        txt_file2=None,
        transform=None,
        audio_transform=None,
        n_frames=8,
        frame_sample_second=2.0,
    ):
        self.name = "vb100"
        self.video_root = Path(video_root)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.audio_transform = audio_transform
        self.n_frames = n_frames
        self.frame_sample_second = frame_sample_second

        # Parse text files
        data = []
        raw_labels = set()
        for txt_file in filter(None, [txt_file1, txt_file2]):
            with open(txt_file, "r") as f:
                for line in f:
                    path, label = line.strip().split()
                    species = Path(path).parent.name.replace("_", " ")
                    raw_label = int(label)
                    raw_labels.add(raw_label)
                    data.append({
                        "path": path,
                        "species": species,
                        "raw_label": raw_label
                    })

        # Map raw labels -> contiguous 0..N-1
        self.raw_to_id = {raw: i for i, raw in enumerate(sorted(raw_labels))}
        self.id2name = {}
        for item in data:
            item["label_id"] = self.raw_to_id[item["raw_label"]]
            self.id2name[item["label_id"]] = item["species"]

        self.df = pd.DataFrame(data)

    def __len__(self):
        return len(self.df)

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
            if not cap.isOpened():
                return Image.new("RGB", (224, 224), color=(0, 0, 0))

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0 or total_frames <= 0:
                cap.release()
                return Image.new("RGB", (224, 224), color=(0, 0, 0))
            
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

    def _extract_frames_uniform(self, video_path, n_frames=8):
        """Extract `n_frames` uniformly spaced frames from the video."""
        frames = []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return [Image.new("RGB", (224, 224), color=(0, 0, 0)) for _ in range(n_frames)]

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return [Image.new("RGB", (224, 224), color=(0, 0, 0)) for _ in range(n_frames)]

        # Compute equally spaced frame indices (middle of each segment)
        segment_length = total_frames / n_frames
        frame_indices = [int(segment_length * (i + 0.5)) for i in range(n_frames)]

        for fid in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            success, frame = cap.read()
            if success and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            else:
                frames.append(Image.new("RGB", (224, 224), color=(0, 0, 0)))
        cap.release()
        return frames

    def _extract_audio(self, video_path):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
            warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
            try:
                waveform, sr = librosa.load(str(video_path), sr=22050, mono=True)
                if waveform is None or len(waveform) == 0:
                    raise ValueError("Empty waveform returned")
            except Exception:
                sr = 22050
                waveform = torch.zeros(1, 220500)
                if self.audio_transform:
                    waveform = self.audio_transform(waveform)
                return waveform, sr
            waveform = torch.tensor(waveform).unsqueeze(0)
            if self.audio_transform:
                waveform = self.audio_transform(waveform)
            return waveform, sr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = self.video_root / row["path"]
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # --- Extract single frame at specified second ---
        frame = self._extract_frame_at_second(video_path, self.frame_sample_second)
        frame_tensor = self.transform(frame)  # single tensor

        # --- Extract audio ---
        waveform, sr = self._extract_audio(video_path)

        # --- Caption ---
        caption = f"an image of a {row['species']}"

        return frame_tensor, waveform, caption, row["label_id"]

    def get_texts(self):
        """Return captions in order of label_id (0..N-1) for zero-shot evaluation."""
        return [f"{self.id2name[i]}" for i in range(len(self.id2name))]

    def get_labels(self):
        """Return contiguous label IDs in order 0..N-1"""
        return list(range(len(self.id2name)))



class ToMono(nn.Module):
    def forward(self, waveform):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform
class PadOrCrop(nn.Module):
    def __init__(self, target_frames: int):
        super().__init__()
        self.target_frames = target_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [channel, n_mels, time]
        n_frames = x.size(-1)
        if n_frames < self.target_frames:
            # pad at the end
            pad_amount = self.target_frames - n_frames
            x = nn.functional.pad(x, (0, pad_amount))
        elif n_frames > self.target_frames:
            # crop
            x = x[..., :self.target_frames]
        return x
