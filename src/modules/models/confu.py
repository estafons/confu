import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.modules.encoders.mlp import MLP
from torch.optim.lr_scheduler import CosineAnnealingLR


def extract_embeddings(model, dataloader, device="cuda:0"):
    """
    Compute embeddings for all samples in a dataloader using a frozen model.
    Returns tuple of projections and labels.
    """
    model.eval()
    emb_1s, emb_2s, emb_3s, labels_list = [], [], [], []
    femb12s, femb13s, femb23s = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            images, audios, texts, labels = [x.to(device) for x in batch]
            femb12, femb13, femb23, emb_1, emb_2, emb_3 = model(images, audios, texts)
            emb_1s.append(emb_1.cpu())
            emb_2s.append(emb_2.cpu())
            emb_3s.append(emb_3.cpu())
            femb12s.append(femb12.cpu())
            femb13s.append(femb13.cpu())
            femb23s.append(femb23.cpu())

            labels_list.append(labels.cpu())

    emb_1s = torch.cat(emb_1s)
    emb_2s = torch.cat(emb_2s)
    emb_3s = torch.cat(emb_3s)
    femb12s = torch.cat(femb12s)
    femb13s = torch.cat(femb13s)
    femb23s = torch.cat(femb23s)
    labelss = torch.cat(labels_list)

    return femb12s, femb13s, femb23s, emb_1s, emb_2s, emb_3s, labelss


class ConFu(pl.LightningModule):
    def __init__(self, modality1_encoder: nn.Module, modality2_encoder: nn.Module, modality3_encoder: nn.Module,
                 embed_dim: int = 256, lr: float = 1e-4, lambda_: float = 0.5, mask_ratio: float = 0.0, fusion_hidden_dim = None, weight_decay: float = 1e-4, fusion_mlp_layers: int = 2):
        super().__init__()

        self.save_hyperparameters()  # saves all __init__ arguments
        self.weight_decay = weight_decay
        self.modality1_encoder = modality1_encoder
        self.modality2_encoder = modality2_encoder
        self.modality3_encoder = modality3_encoder
        self.lambda_ = lambda_
        self.mask_ratio = mask_ratio
        self.fusion_hidden_dim = fusion_hidden_dim
        self.fusion_mlp_layers = fusion_mlp_layers

        # projections into shared space
        self.mod12_fusion_head = MLP(
                indim=modality1_encoder.output_dim + modality2_encoder.output_dim, hiddim=self.fusion_hidden_dim, outdim=embed_dim, dropout=True, dropoutp=0.1, num_layers=self.fusion_mlp_layers)
        self.mod13_fusion_head = MLP(
                indim=modality1_encoder.output_dim + modality3_encoder.output_dim, hiddim=self.fusion_hidden_dim, outdim=embed_dim, dropout=True, dropoutp=0.1, num_layers=self.fusion_mlp_layers)
        self.mod23_fusion_head = MLP(
                indim=modality2_encoder.output_dim + modality3_encoder.output_dim, hiddim=self.fusion_hidden_dim, outdim=embed_dim, dropout=True, dropoutp=0.1, num_layers=self.fusion_mlp_layers)
        self.mod1_proj = nn.Linear(modality1_encoder.output_dim, embed_dim)
        self.mod2_proj = nn.Linear(modality2_encoder.output_dim, embed_dim)
        self.mod3_proj = nn.Linear(modality3_encoder.output_dim, embed_dim)

        self.lr = lr
        
    def forward(self, image_inputs, audio_inputs, text_inputs):
        # encode
      #  print(f"Image inputs shape: {image_inputs.shape}, Audio inputs shape: {audio_inputs.shape}, Text inputs shape: {text_inputs.shape}")
        
        img_emb, _ = self.modality1_encoder(image_inputs)         # [B, D_img]
        aud_emb, _ = self.modality2_encoder(audio_inputs)         # [B, D_aud]
        txt_emb, _ = self.modality3_encoder(text_inputs)  # [B, D_txt]


        mod1_emb = self.mod1_proj(img_emb)  # [B, embed_dim]
        mod2_emb = self.mod2_proj(aud_emb)  # [B, embed_dim]
        mod3_emb = self.mod3_proj(txt_emb)  # [B, embed

        mod1_emb = F.normalize(mod1_emb, dim=-1)  # [B, embed_dim]
        mod2_emb = F.normalize(mod2_emb, dim=-1)  # [B, embed_dim]
        mod3_emb = F.normalize(mod3_emb, dim=-1)

        if self.lambda_ == 0.0:
            # no fusion terms
            return mod1_emb, mod2_emb, mod3_emb, mod1_emb, mod2_emb, mod3_emb

        to_mask_img = img_emb
        to_mask_aud = aud_emb
        to_mask_txt = txt_emb
        if self.mask_ratio > 0 and self.training:
            
            mask_12 = torch.rand(to_mask_img.shape, device=self.device) < self.mask_ratio
            mask_13 = torch.rand(to_mask_aud.shape, device=self.device) < self.mask_ratio
            mask_23 = torch.rand(to_mask_txt.shape, device=self.device) < self.mask_ratio

            img_emb_m = to_mask_img * mask_12
            aud_emb_m = to_mask_aud * mask_23
            txt_emb_m = to_mask_txt * mask_13
        else:
            img_emb_m = to_mask_img
            aud_emb_m = to_mask_aud
            txt_emb_m = to_mask_txt

        mod12_emb = torch.cat([img_emb_m, aud_emb_m], dim=-1)
        mod13_emb = torch.cat([img_emb_m, txt_emb_m], dim=-1)
        mod23_emb = torch.cat([aud_emb_m, txt_emb_m], dim=-1)

        

        mod12_emb = self.mod12_fusion_head(mod12_emb)  # [B, embed_dim]
        mod13_emb = self.mod13_fusion_head(mod13_emb)  # [B, embed_dim]
        mod23_emb = self.mod23_fusion_head(mod23_emb)

        mod12_proj = F.normalize(mod12_emb, dim=-1)    # [B, embed_dim]
        mod13_proj = F.normalize(mod13_emb, dim=-1)    # [B, embed_dim]
        mod23_proj = F.normalize(mod23_emb, dim=-1)    # [B, embed_dim]


        return mod12_proj, mod13_proj, mod23_proj, mod1_emb, mod2_emb, mod3_emb

    def video_forward(self, image_inputs, audio_inputs, text_inputs):
        mod12_proj_list = []
        mod13_proj_list = []
        mod1_proj_list = []
        for image_input in image_inputs:
            
            image_input = image_input.to(self.device)
            mod12_proj, mod13_proj, mod23_proj, mod1_emb, mod2_emb, mod3_emb = self.forward(image_input, audio_inputs, text_inputs)
            mod12_proj_list.append(mod12_proj)
            mod13_proj_list.append(mod13_proj)
            mod1_proj_list.append(mod1_emb)
        # average over time dimension
        mod12_proj = torch.mean(torch.stack(mod12_proj_list, dim=0), dim=0)
        mod13_proj = torch.mean(torch.stack(mod13_proj_list, dim=0), dim=0)
        mod1_emb = torch.mean(torch.stack(mod1_proj_list, dim=0), dim=0)
        mod12_proj = F.normalize(mod12_proj, dim=-1)
        mod13_proj = F.normalize(mod13_proj, dim=-1)
        mod1_emb = F.normalize(mod1_emb, dim=-1)
        return mod12_proj, mod13_proj, mod23_proj, mod1_emb, mod2_emb, mod3_emb

    def contrastive_loss(self, ia_proj, txt_proj, temperature: float = 0.07):
        logits = ia_proj @ txt_proj.t() / temperature  # [B, B]
        labels = torch.arange(logits.size(0), device=self.device)

        loss_i2t = F.cross_entropy(logits, labels)         # image+audio -> text
        loss_t2i = F.cross_entropy(logits.t(), labels)     # text -> image+audio

        return (loss_i2t + loss_t2i) / 2

    def pairwise_contrastive_loss(self, mod12_proj, mod13_proj, mod23_proj, mod1_emb, mod2_emb, mod3_emb, temperature: float = 0.07):
        cl12 = self.contrastive_loss(mod1_emb, mod2_emb, temperature)
        cl23 = self.contrastive_loss(mod2_emb, mod3_emb, temperature)
        cl31 = self.contrastive_loss(mod3_emb, mod1_emb, temperature)
        
        if self.lambda_ == 0.0:
            return (cl12 + cl23 + cl31) / 3, (None, None, None, cl12, cl23, cl31)
        clf1 = self.contrastive_loss(mod12_proj, mod3_emb, temperature)
        clf2 = self.contrastive_loss(mod13_proj, mod2_emb, temperature)
        clf3 = self.contrastive_loss(mod23_proj, mod1_emb, temperature)
        
        
        return (self.lambda_ * (clf1 + clf2 + clf3) + (1 - self.lambda_) * (cl12 + cl23 + cl31)) / 6, (clf1, clf2, clf3, cl12, cl23, cl31)

    def training_step(self, batch, batch_idx):
        images, audios, texts, *rest = batch

        mod12_proj, mod13_proj, mod23_proj, mod1_emb, mod2_emb, mod3_emb = self(images, audios, texts)
        loss, losses_ = self.pairwise_contrastive_loss(mod12_proj, mod13_proj, mod23_proj, mod1_emb, mod2_emb, mod3_emb)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, audios, texts, *rest = batch

        mod12_proj, mod13_proj, mod23_proj, mod1_emb, mod2_emb, mod3_emb = self(images, audios, texts)
        loss, _ = self.pairwise_contrastive_loss(mod12_proj, mod13_proj, mod23_proj, mod1_emb, mod2_emb, mod3_emb)
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
        
        # Only include fusion parameters if lambda > 0
        if self.lambda_ > 0.0:
            params.extend([
                {'params': self.mod12_fusion_head.parameters()},
                {'params': self.mod13_fusion_head.parameters()},
                {'params': self.mod23_fusion_head.parameters()},
            ])
            
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
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
    

