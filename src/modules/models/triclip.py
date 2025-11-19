import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR

def extract_embeddings(model, dataloader, device="cuda:0"):
    """
    Compute embeddings for all samples in a dataloader using a frozen model.
    Returns tuple of projections and labels.
    """
    model.eval()
    emb_1s, emb_2s, emb_3s, labels_list = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            images, audios, texts, labels = [x.to(device) for x in batch]
            emb_1, emb_2, emb_3 = model(images, audios, texts)
            emb_1s.append(emb_1.cpu())
            emb_2s.append(emb_2.cpu())
            emb_3s.append(emb_3.cpu())
            

            labels_list.append(labels.cpu())

    emb_1s = torch.cat(emb_1s)
    emb_2s = torch.cat(emb_2s)
    emb_3s = torch.cat(emb_3s)

    labels = torch.cat(labels_list)

    return emb_1s, emb_2s, emb_3s, labels


class ThreeModalityBaselineCLIPModule(pl.LightningModule):
    def __init__(self, modality1_encoder: nn.Module, modality2_encoder: nn.Module, modality3_encoder: nn.Module,
                 embed_dim: int = 256, lr: float = 1e-4, weight_decay: float = 1e-4):
        super().__init__()

        self.save_hyperparameters()  # saves all __init__ arguments

        self.modality1_encoder = modality1_encoder
        self.modality2_encoder = modality2_encoder
        self.modality3_encoder = modality3_encoder
        

        self.mod1_proj = nn.Linear(modality1_encoder.output_dim, embed_dim)
        self.mod2_proj = nn.Linear(modality2_encoder.output_dim, embed_dim)
        self.mod3_proj = nn.Linear(modality3_encoder.output_dim, embed_dim)

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, image_inputs, audio_inputs, text_inputs):
        # encode
      #  print(f"Image inputs shape: {image_inputs.shape}, Audio inputs shape: {audio_inputs.shape}, Text inputs shape: {text_inputs.shape}")
        img_emb, _ = self.modality1_encoder(image_inputs)         # [B, D_img]
        aud_emb, _ = self.modality2_encoder(audio_inputs)         # [B, D_aud]
        txt_emb, _ = self.modality3_encoder(text_inputs)  # [B, D_txt]

        # fuse image + audio
      # ia_emb = torch.cat([img_emb, aud_emb], dim=-1)     # [B, D_img + D_aud]
        # fuse pairwise


        mod1_emb = self.mod1_proj(img_emb)  # [B, embed_dim]
        mod2_emb = self.mod2_proj(aud_emb)  # [B, embed_dim]
        mod3_emb = self.mod3_proj(txt_emb)  # [B, embed

        mod1_emb = F.normalize(mod1_emb, dim=-1)  # [B, embed_dim]
        mod2_emb = F.normalize(mod2_emb, dim=-1)  # [B, embed_dim]
        mod3_emb = F.normalize(mod3_emb, dim=-1)

        # project to common space
        # ia_proj = F.normalize(self.ia_proj(ia_emb), dim=-1)    # [B, embed_dim]
        # txt_proj = F.normalize(self.txt_proj(txt_emb), dim=-1) # [B, embed_dim]

        return mod1_emb, mod2_emb, mod3_emb

    def video_forward(self, image_inputs, audio_inputs, text_inputs):

        image_embs = []

        for image in image_inputs:
            image = image.to(self.device)
            mod1_emb, mod2_emb, mod3_emb = self(image, audio_inputs, text_inputs)
            image_embs.append(mod1_emb)

        mod1_emb = torch.stack(image_embs, dim=0).mean(dim=0)

        return mod1_emb, mod2_emb, mod3_emb


    def contrastive_loss(self, mod1_proj, mod2_proj, temperature: float = 0.07):
        logits = mod1_proj @ mod2_proj.t() / temperature  # [B, B]
        labels = torch.arange(logits.size(0), device=self.device)

        loss_m1_to_m2 = F.cross_entropy(logits, labels)         # modality1 -> modality2
        loss_m2_to_m1 = F.cross_entropy(logits.t(), labels)     # modality2 -> modality1

        return (loss_m1_to_m2 + loss_m2_to_m1) / 2

    def pairwise_contrastive_loss(self, mod1_emb, mod2_emb, mod3_emb, temperature: float = 0.07):
        cl1 = self.contrastive_loss(mod1_emb, mod3_emb, temperature)
        cl2 = self.contrastive_loss(mod2_emb, mod3_emb, temperature)
        cl3 = self.contrastive_loss(mod1_emb, mod2_emb, temperature)
        return (cl1 + cl2 + cl3) / 3
        

    def training_step(self, batch, batch_idx):
        images, audios, texts, *rest = batch

        mod1_emb, mod2_emb, mod3_emb = self(images, audios, texts)
        loss = self.pairwise_contrastive_loss(mod1_emb, mod2_emb, mod3_emb)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, audios, texts, *rest = batch

        mod1_emb, mod2_emb, mod3_emb = self(images, audios, texts)
        loss = self.pairwise_contrastive_loss(mod1_emb, mod2_emb, mod3_emb)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    

    def configure_optimizers(self):
        params = [
            {'params': self.modality1_encoder.parameters()},
            {'params': self.modality2_encoder.parameters()},
            {'params': self.modality3_encoder.parameters()},
            {'params': self.mod1_proj.parameters()},
            {'params': self.mod2_proj.parameters()},
            {'params': self.mod3_proj.parameters()}
        ]
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)

        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
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
    

