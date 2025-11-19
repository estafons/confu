
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch
from torchvision import transforms
import torch.nn as nn
from src.datasets.ssw60_eval_datamodule import BirdVideoDatasetSingleFrame, BirdVideoDataset
from src.datasets.vb100_dataset import VB100Dataset, VB100DatasetSingleFrame
from src.datasets.cub_dataset import CUBBirdsDataset


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

class ToMono(nn.Module):
    def forward(self, waveform):
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform

def get_dataset(cfg, which='ssw60', audio_transform='default', transform='default', single_frame_eval=False, sample_sec=2.0):
     # --- Create dataset ---
    if transform == 'default':
        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
    else:
        transform = transform  # use provided transform

    if audio_transform == 'default':
        audio_transform = torch.nn.Sequential(
            ToMono(),
            MelSpectrogram(
                sample_rate=22050,
                n_fft=1024,           # window size (~46ms)
                hop_length=512,       # step size (~23ms)
                n_mels=128,           # number of Mel bands
                f_min=500,            # optional: focus on 500Hz-10kHz, typical bird vocal range
                f_max=10000
            ),
            PadOrCrop(target_frames=431),  # ensure fixed length of ~10s
            AmplitudeToDB()  # convert to log scale
        )
    else:
        audio_transform = audio_transform  # use provided transform
    if which == 'ssw60':
        
        if single_frame_eval:
            dataset = BirdVideoDatasetSingleFrame(
                paired_csv=cfg.ssw60_videos_csv,
                taxa_csv=cfg.ssw60_taxa_csv,
                video_dir=cfg.ssw60_video_dir,
                transform=transform,
                audio_transform=audio_transform,
                frame_sample_second=sample_sec
            )
        else:
            dataset = BirdVideoDataset(
                paired_csv=cfg.ssw60_videos_csv,
                taxa_csv=cfg.ssw60_taxa_csv,
                video_dir=cfg.ssw60_video_dir,
                transform=transform,
                audio_transform=audio_transform,
            )
    elif which == 'vb100':
        if single_frame_eval:
            dataset = VB100DatasetSingleFrame(
                txt_file1=cfg.vb100_train_csv,
                video_root=cfg.vb100_videos_dir,
                txt_file2=cfg.vb100_test_csv,
                transform=transform,
                audio_transform=audio_transform,
                frame_sample_second=sample_sec
            )
        else:
            dataset = VB100Dataset(
                txt_file1=cfg.vb100_train_csv,
                video_root=cfg.vb100_videos_dir,
                txt_file2=cfg.vb100_test_csv,
                transform=transform,
                audio_transform=audio_transform,
            )
    elif which == 'cub':
        dummy_text = "dummy multimodal input"
        dummy_audios = torch.randn(1, 128, 431)
        dataset = CUBBirdsDataset(
            root_dir=cfg.cub_root_dir,
            transform=transform,
            dummy_audio=dummy_audios,
            dummy_text=dummy_text
        )

    return dataset