import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.modules.encoders.resnet import create_image_encoder, create_audio_encoder, create_text_encoder
from src.modules.models.confu import ConFu
from src.datasets.bird_mml import BirdCaptionDataModule
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import pandas as pd
import hydra

import torch
import pandas as pd




@hydra.main(config_path="../../../configs", config_name="birds")
def run_confu_birdtriplet(cfg):
    embed_dim = cfg.embed_dim

    # --- Create encoders ---
    image_encoder = create_image_encoder(embed_dim=embed_dim, in_channels=3, backbone='resnet50')
    audio_encoder = create_audio_encoder(embed_dim=embed_dim, in_channels=1, backbone='resnet50')
    text_encoder = create_text_encoder(embed_dim=embed_dim)

    # --- Initialize FusionCLIPModule2 ---
    fusionclip = ConFu(
        modality1_encoder=image_encoder,
        modality2_encoder=audio_encoder,
        modality3_encoder=text_encoder,
        embed_dim=embed_dim,
        lr=1e-4,
        lambda_=0.5,
        mask_ratio=0.0,
        fusion_hidden_dim=512,
        weight_decay=1e-4,
    )
   
    print("ConFu initialized successfully!")

    # --- Setup BirdTripletDataModule ---
    # audio_csv = cfg.audio_csv
    
    # optional transforms
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    audio_transform = torch.nn.Sequential(
        MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,           # window size (~46ms)
            hop_length=512,       # step size (~23ms)
            n_mels=128,           # number of Mel bands
            f_min=500,            # optional: focus on 500Hz-10kHz, typical bird vocal range
            f_max=10000
        ),
        AmplitudeToDB()  # convert to log scale
    )

  
    dm = BirdCaptionDataModule(
        train_csv=cfg.train_csv,
        test_csv=cfg.audio_csv,
        taxa_csv=cfg.species_csv,
        audio_dir=cfg.audio_dir,
        image_sources=cfg.image_sources,
        batch_size=cfg.batch_size,
        transform=transform,
        audio_transform=audio_transform
    )
    dm.setup()

        # --- Trainer --- save last model
    # --- Trainer --- save last model + top-K
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="checkpoint-{epoch:02d}-{step}",
        save_top_k=-1,      # save all checkpoints
        save_last=True,
        monitor=None,       # no monitored metric
        verbose=True,
        every_n_epochs=20     # ✅ save every 20 epochs
    )


    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback]
    )

    # --- Train ---
    trainer.fit(fusionclip, dm) 


if __name__ == "__main__":
    run_confu_birdtriplet()