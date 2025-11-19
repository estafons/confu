from src.modules.models.triclip import ThreeModalityBaselineCLIPModule
from src.modules.encoders.resnet import create_audio_encoder, create_text_encoder, create_image_encoder

from src.datasets.av_mnist_datamodule import AVMNISTDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
import pytorch_lightning as pl
import os
from src.experiments.av_mnist.utils import save_results
import torch
import torch.nn.functional as F

@hydra.main(config_path="../../../configs", config_name="av_mnist")
def run_trimodal_clip_avmnist(cfg):
    embed_dim = cfg.embed_dim
    avmnist_file_path = cfg.data_path

    # create encoders
    image_encoder = create_image_encoder(embed_dim=embed_dim)
    audio_encoder = create_audio_encoder(embed_dim=embed_dim)
    text_encoder = create_text_encoder(embed_dim=embed_dim)

    # initialize ThreeModalityBaselineCLIPModule
    model = ThreeModalityBaselineCLIPModule(
        modality1_encoder=image_encoder,
        modality2_encoder=audio_encoder,
        modality3_encoder=text_encoder,
        embed_dim=embed_dim
    )

    # setup data
    dm = AVMNISTDataModule(batch_size=cfg.batch_size, data_path=avmnist_file_path)
    dm.prepare_data()
    dm.setup()

    # checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        filename='best_trimodal_model',
        save_top_k=1,
        mode='min',
        verbose=True
    )

    # trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
    )

    # training
    trainer.fit(model, dm)

    # load best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    model = ThreeModalityBaselineCLIPModule.load_from_checkpoint(
        best_model_path,
        modality1_encoder=image_encoder,
        modality2_encoder=audio_encoder,
        modality3_encoder=text_encoder,
        embed_dim=embed_dim
    )
    model.eval()

    # --- Zero-shot evaluation using image-text ---
    texts = [
        "an image of a zero",
        "an image of a one",
        "an image of a two",
        "an image of a three",
        "an image of a four",
        "an image of a five",
        "an image of a six",
        "an image of a seven",
        "an image of an eight",
        "an image of a nine",
    ]
    # encode texts
    text_embs, _ = model.modality3_encoder(texts)
    text_embs = F.normalize(model.mod3_proj(text_embs), dim=-1)
    correct_audio = 0
    correct_image = 0
    total = 0
    test_loader = dm.test_dataloader()
    with torch.no_grad():
        for images, audios, texts_batch, labels in test_loader:
            images = images.to(model.device)
            audios = audios.to(model.device)
            labels = labels.to(model.device)

            # encode images and audio
            image_emb, audio_emb, _ = model(images, audios, texts_batch)

            # use image embeddings for zero-shot image-text evaluation
            logits_per_image = image_emb @ text_embs.t()

           
            
            probs = logits_per_image.softmax(dim=-1)
            preds = probs.argmax(dim=-1)
            correct_image += (preds == labels).sum().item()
            total += labels.size(0)

             # use audio embeddings for zero-shot audio-text evaluation
            logits_per_audio = audio_emb @ text_embs.t()

            probs = logits_per_audio.softmax(dim=-1)
            preds = probs.argmax(dim=-1)
            correct_audio += (preds == labels).sum().item()

    print(f"Zero-shot classification accuracy (audio-text): {correct_audio / total:.4f}")

    print(f"Zero-shot classification accuracy (image-text): {correct_image / total:.4f}")

    save_path = os.path.join(cfg.results_path, "avmnist_trimodal_results.csv")
    save_results(save_path=save_path, iteration=cfg.iteration, correct_audio=correct_audio, correct_image=correct_image, total=total)


if __name__ == "__main__":
    run_trimodal_clip_avmnist()
