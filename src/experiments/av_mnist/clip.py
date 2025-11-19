from src.datasets.av_mnist_datamodule import AVMNISTDataModule
from src.modules.encoders.resnet import create_audio_encoder, create_text_encoder, create_image_encoder
from src.modules.models.clip import DualCLIPModule
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
import os
from src.experiments.av_mnist.utils import save_results
import hydra

@hydra.main(config_path="../../../configs", config_name="av_mnist")
def run_dual_clip_avmnist(cfg):
    embed_dim = cfg.embed_dim
    avmnist_file_path = cfg.data_path
    dm = AVMNISTDataModule(batch_size=cfg.batch_size, data_path=avmnist_file_path)
    dm.prepare_data()
    dm.setup()

    # --- Audio-Text CLIP ---
    audio_encoder = create_audio_encoder(embed_dim=embed_dim)
    text_encoder = create_text_encoder(embed_dim=embed_dim)
    audio_text_model = DualCLIPModule(audio_encoder=audio_encoder, text_encoder=text_encoder, mode="audio-text")

    checkpoint_audio = ModelCheckpoint(
        monitor='val_loss', filename='best_audio_text', save_top_k=1, mode='min', verbose=True
    )

    trainer = pl.Trainer(max_epochs=cfg.max_epochs, accelerator="auto", devices="auto", callbacks=[checkpoint_audio])
    trainer.fit(audio_text_model, dm)

    # --- Image-Text CLIP ---
    image_encoder = create_image_encoder(embed_dim=embed_dim)
    text_encoder = create_text_encoder(embed_dim=embed_dim)  # separate instance
    image_text_model = DualCLIPModule(image_encoder=image_encoder, text_encoder=text_encoder, mode="image-text")

    checkpoint_image = ModelCheckpoint(
        monitor='val_loss', filename='best_image_text', save_top_k=1, mode='min', verbose=True
    )

    trainer = pl.Trainer(max_epochs=cfg.max_epochs, accelerator="auto", devices="auto", callbacks=[checkpoint_image])
    trainer.fit(image_text_model, dm)

    # load best models
    best_model_path_audio = checkpoint_audio.best_model_path
    print(f"Best model saved at: {best_model_path_audio}")
    audio_text_model = DualCLIPModule.load_from_checkpoint(
        best_model_path_audio,
        audio_encoder=audio_encoder,
        text_encoder=text_encoder,
        mode="audio-text"
    )
    audio_text_model.eval()
    best_model_path_image = checkpoint_image.best_model_path
    print(f"Best model saved at: {best_model_path_image}")
    image_text_model = DualCLIPModule.load_from_checkpoint(
        best_model_path_image,
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        mode="image-text"
    )

    # --- Zero-Shot Evaluation ---
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
    text_emb_audio, _ = audio_text_model.text_encoder(texts)
    text_emb_image, _ = image_text_model.text_encoder(texts)

    correct_audio, total = 0, 0
    correct_image = 0
    test_loader = dm.test_dataloader()
    with torch.no_grad():
        for images, audios, texts_batch, labels in test_loader:
            images = images.to(image_text_model.device)
            audios = audios.to(audio_text_model.device)
            labels = labels.to(image_text_model.device)

            # Audio-Text
            audio_emb, _ = audio_text_model(audios=audios, texts=texts_batch)
            logits_audio = audio_emb @ text_emb_audio.t()
            preds_audio = logits_audio.argmax(dim=-1)
            correct_audio += (preds_audio == labels).sum().item()
            total += labels.size(0)

            # Image-Text
            image_emb, _ = image_text_model(images=images, texts=texts_batch)
            logits_image = image_emb @ text_emb_image.t()
            preds_image = logits_image.argmax(dim=-1)
            correct_image += (preds_image == labels).sum().item()
           

    print(f"Zero-shot accuracy (audio-text): {correct_audio / total:.4f}")
    print(f"Zero-shot accuracy (image-text): {correct_image / total:.4f}")
    save_path = os.path.join(cfg.results_path, "avmnist_dual_clip_results.csv")
    save_results(save_path=save_path, iteration=cfg.iteration, correct_audio=correct_audio, correct_image=correct_image, total=total)



if __name__ == "__main__":
    run_dual_clip_avmnist()