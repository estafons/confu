import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.functional import normalize

class DualCLIPModule(pl.LightningModule):
    """
    Trains either audio-text or image-text alignment, depending on mode.
    mode = 'audio-text' or 'image-text'
    """
    def __init__(self, image_encoder=None, audio_encoder=None, text_encoder=None,
                 embed_dim=256, lr=1e-4, weight_decay=1e-4, mode="image-text"):
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim
        self.lr = lr
        self.weight_decay = weight_decay

        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

        if mode == "image-text":
            self.image_encoder = image_encoder
        elif mode == "audio-text":
            self.audio_encoder = audio_encoder
        else:
            raise ValueError("Mode must be 'image-text' or 'audio-text'")

        self.text_encoder = text_encoder

    def forward(self, images=None, audios=None, texts=None):
        text_emb, _ = self.text_encoder(texts)
        text_emb = normalize(text_emb, dim=-1)

        if self.mode == "image-text":
            image_emb, _ = self.image_encoder(images)
            image_emb = normalize(image_emb, dim=-1)
            return image_emb, text_emb
        else:  # audio-text
            audio_emb, _ = self.audio_encoder(audios)
            audio_emb = normalize(audio_emb, dim=-1)
            return audio_emb, text_emb

    def video_forward(self, images=None, audios=None, texts=None):
        image_embs = []
        for image in images:
            image = image.to(self.device)
            image_emb, text_emb = self.forward(images=image, texts=texts)
            image_embs.append(image_emb)
        image_emb = torch.mean(torch.stack(image_embs, dim=0), dim=0)

        image_emb = normalize(image_emb, dim=-1)
        return image_emb, text_emb


    def training_step(self, batch, batch_idx):
        images, audios, texts, _ = batch
        if self.mode == "image-text":
            image_emb, text_emb = self(images=images, texts=texts)
            logits = image_emb @ text_emb.t() * self.logit_scale.exp()
        else:
            audio_emb, text_emb = self(audios=audios, texts=texts)
            logits = audio_emb @ text_emb.t() * self.logit_scale.exp()

        labels = torch.arange(len(logits), device=self.device)
        loss = (nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.t(), labels)) / 2
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, audios, texts, _ = batch
        if self.mode == "image-text":
            image_emb, text_emb = self(images=images, texts=texts)
            logits = image_emb @ text_emb.t() * self.logit_scale.exp()
        else:
            audio_emb, text_emb = self(audios=audios, texts=texts)
            logits = audio_emb @ text_emb.t() * self.logit_scale.exp()

        labels = torch.arange(len(logits), device=self.device)
        loss = (nn.CrossEntropyLoss()(logits, labels) + nn.CrossEntropyLoss()(logits.t(), labels)) / 2
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        
       # return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Define cosine annealing scheduler
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,  # period (in epochs)
            eta_min=1e-6                    # minimum learning rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # or "step" for per-batch updates
                "frequency": 1,
                "monitor": "val_loss",  # optional, for ReduceLROnPlateau
            },
        }

