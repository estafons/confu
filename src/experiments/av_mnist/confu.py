
from src.modules.encoders.resnet import create_audio_encoder, create_text_encoder, create_image_encoder
from src.modules.models.confu import ConFu
from src.datasets.av_mnist_datamodule import AVMNISTDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import torch
import os
import hydra
from src.experiments.av_mnist.utils import save_results

@hydra.main(config_path="../../../configs", config_name="av_mnist")
def run_confu_avmnist(cfg):
    embed_dim = cfg.embed_dim
    avmnist_file_path = cfg.data_path
    image_encoder = create_image_encoder(embed_dim=embed_dim)
    audio_encoder = create_audio_encoder(embed_dim=embed_dim)
    text_encoder = create_text_encoder(embed_dim=embed_dim)

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
    
    dm = AVMNISTDataModule(batch_size=cfg.batch_size, data_path=avmnist_file_path)
    dm.prepare_data()
    dm.setup()

    # train
    checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',  # make sure val_loss is logged in your LightningModule
            filename='best_model',
            save_top_k=1,
            mode='min',
            verbose=True
        )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
    )

    # Train
    trainer.fit(fusionclip, dm)

    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    fusionclip = ConFu.load_from_checkpoint(best_model_path, modality1_encoder=image_encoder, modality2_encoder=audio_encoder, modality3_encoder=text_encoder)

    print("Model loaded successfully. Now extracting embeddings...")

    # Extract embeddings
    dm.setup()
    test_loader = dm.test_dataloader()


    # evaluate on zero shot classification

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
    dummy_images = torch.randn((10, 1, 28, 28)).to(fusionclip.device)
    dummy_audios = torch.randn((10, 1, 112, 112)).to(fusionclip.device)
    correct = 0
    total = 0
    correct_image = 0
    correct_audio = 0
    fusionclip.eval() 
    text_embs = fusionclip(dummy_images, dummy_audios, texts)[-1]
     # ADD THIS LINE - Set model to evaluation mode
    with torch.no_grad():
        for batch in test_loader:
            images, audios, texts, labels = batch
            images = images.to(fusionclip.device)
            audios = audios.to(fusionclip.device)
            image_audio, image_text, audio_text, image, audio, text = fusionclip(
                images, audios, texts
            )
            print("Image Embeddings Shape:", image.shape)
            print("Audio Embeddings Shape:", audio.shape)
            print("Text Embeddings Shape:", text.shape)

            # compute similarity
            logits_per_image_audio = image_audio @ text_embs.t()
            # print("Logits per image shape:", logits_per_image.shape)
            # print(logits_per_image)
            probs = logits_per_image_audio.softmax(dim=-1)  # (batch_size, num_texts)
            preds = probs.argmax(dim=-1)
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)

            logits_per_image = image @ text_embs.t()
            probs_simple = logits_per_image.softmax(dim=-1)
            preds_simple = probs_simple.argmax(dim=-1)
            correct_image += (preds_simple.cpu() == labels).sum().item()

            logits_per_audio = audio @ text_embs.t()
            probs_audio = logits_per_audio.softmax(dim=-1)
            preds_audio = probs_audio.argmax(dim=-1)
            correct_audio += (preds_audio.cpu() == labels).sum().item()

    print(f"Zero-shot classification accuracy: {correct / total:.4f}")
    print(f"Zero-shot classification accuracy (image): {correct_image / total:.4f}")
    print(f"Zero-shot classification accuracy (audio): {correct_audio / total:.4f}")

    # append results to csv file
    save_path = os.path.join(cfg.results_path, "avmnist_confu_results.csv")
    save_results(save_path=save_path, iteration=cfg.iteration, correct_audio=correct_audio, correct_image=correct_image, total=total, correct_audio_visual=correct)



if __name__ == "__main__":
    run_confu_avmnist()