
from src.modules.models.triangle import TiangleBaselineCLIPModule
from src.modules.encoders.resnet import create_audio_encoder, create_text_encoder, create_image_encoder

from src.datasets.av_mnist_datamodule import AVMNISTDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
import pytorch_lightning as pl
import os
from src.experiments.av_mnist.utils import save_results
import torch

@hydra.main(config_path="../../../configs", config_name="av_mnist")
def run_triangle_baseline_avmnist(cfg):

    embed_dim = cfg.embed_dim
    avmnist_file_path = cfg.data_path
    
    image_encoder = create_image_encoder(embed_dim=embed_dim)
    audio_encoder = create_audio_encoder(embed_dim=embed_dim)
    text_encoder = create_text_encoder(embed_dim=embed_dim)

    triangle_baseline = TiangleBaselineCLIPModule(
        modality1_encoder=image_encoder,
        modality2_encoder=audio_encoder,
        modality3_encoder=text_encoder
    )
    print("TiangleBaselineCLIPModule initialized successfully!")
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
    trainer.fit(triangle_baseline, dm)

    # Load best model
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    triangle_baseline = TiangleBaselineCLIPModule.load_from_checkpoint(best_model_path, modality1_encoder=image_encoder, modality2_encoder=audio_encoder, modality3_encoder=text_encoder)
    print("Model loaded successfully. Now extracting embeddings...")

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
    dummy_images = torch.randn((10, 1, 28, 28)).to(triangle_baseline.device)
    dummy_audios = torch.randn((10, 1, 112, 112)).to(triangle_baseline.device)
    correct = 0
    total = 0
    triangle_baseline.eval()
    text_embs = triangle_baseline(dummy_images, dummy_audios, texts)[-1]
    
    
    with torch.no_grad():
        for batch in test_loader:
            images, audios, texts, labels = batch
            images = images.to(triangle_baseline.device)
            audios = audios.to(triangle_baseline.device)
            image_emb, audio_emb, text_emb = triangle_baseline(
                images, audios, texts
            )
            print("Image Embeddings Shape:", image_emb.shape)
            print("Audio Embeddings Shape:", audio_emb.shape)
            print("Text Embeddings Shape:", text_emb.shape)

            # compute similarity

            
            similarity_scores = -triangle_baseline.area_computation(text_embs, image_emb, audio_emb).T
            print(similarity_scores.shape)
            probs = similarity_scores.softmax(dim=-1)  # (batch_size, num_texts)
            preds = probs.argmax(dim=-1)
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)
            
            
    print(f"Zero-shot classification accuracy: {correct / total:.4f}")
    

    save_path = os.path.join(cfg.results_path, "avmnist_triangle_results.csv")
    save_results(save_path=save_path, iteration=cfg.iteration, total=total, correct_audio_visual=correct)


if __name__ == "__main__":
    run_triangle_baseline_avmnist()