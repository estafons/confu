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

class TiangleBaselineCLIPModule(pl.LightningModule):
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


    
    def area_computation(self, language, video, audio):


    #print(f"norm language= {torch.sum(language ** 2, dim=1)}")
        
        language_expanded = language.unsqueeze(1)  # Shape: (n, 1, dim)

        # Compute the differences for all pairs (i-th language embedding with all j-th video/audio embeddings)
        u = language_expanded - video.unsqueeze(0)  # Shape: (n, n, dim)
        v = language_expanded - audio.unsqueeze(0)  # Shape: (n, n, dim)

        # Compute the norms for u and v
        u_norm = torch.sum(u ** 2, dim=2)  # Shape: (n, n)
        v_norm = torch.sum(v ** 2, dim=2)  # Shape: (n, n)

        # Compute the dot products for all pairs
        uv_dot = torch.sum(u * v, dim=2)  # Shape: (n, n)

        # Calculate the area for all pairs. I remove sqrt calculation
        area = ((u_norm * v_norm) - (uv_dot ** 2))/2#torch.sqrt((u_norm * v_norm) - (uv_dot ** 2)) / 2  # Shape: (n, n)
        
        return area


    def compute_area_loss(self, language, video, audio):
        """
        Implements the same loss as the snippet you provided.
        """
        bs = language.size(0)
        contrastive_temp = 0.07

        # Compute area and areaT
        area = self.area_computation(language, video, audio)
        area = area / contrastive_temp

        areaT = self.area_computation(language, video, audio).T
        areaT = areaT / contrastive_temp

        targets = torch.arange(bs, dtype=torch.long, device=self.device)

        loss = (
            F.cross_entropy(-area, targets, label_smoothing=0.1)
            + F.cross_entropy(-areaT, targets, label_smoothing=0.1)
        ) / 2

        return loss

        

    def training_step(self, batch, batch_idx):
        images, audios, texts, *rest = batch

        mod1_emb, mod2_emb, mod3_emb = self(images, audios, texts)
        loss = self.compute_area_loss(mod1_emb, mod2_emb, mod3_emb)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, audios, texts, *rest = batch

        mod1_emb, mod2_emb, mod3_emb = self(images, audios, texts)
        loss1_23 = self.compute_area_loss(mod1_emb, mod2_emb, mod3_emb)
        # loss2_13 = self.compute_area_loss(mod2_emb, mod1_emb, mod3_emb)
        # loss3_11 = self.compute_area_loss(mod3_emb, mod1_emb, mod2_emb)
        loss = loss1_23  #+ loss2_13 + loss3_11

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
    

    def recall_at_k(self, emb1, emb2, emb3, ks=[1, 5, 10], modalities=[1, 2]):
        
        similarity_scores = -self.area_computation(emb1, emb2, emb3)
        
        return recall_from_sims(similarity_scores, ks=ks, modalities=modalities)



def recall_from_sims(sims, ks=[1,5,10], modalities=None):
    """
    Compute recall@K for cross-modal retrieval both directions.
    Args:
        sims (torch.Tensor): shape [N, N]
        ks (list[int]): list of k values to compute recall@k
    Returns:
        dict: {"m1->m2_recall@k": val, "m2->m1_recall@k": val}
    """
    
    N = sims.size(0)

    results = {}

    # m1 -> m2 retrieval
    ranks = sims.argsort(dim=1, descending=True)  # [N, N]
    gt = torch.arange(N).unsqueeze(1)  # ground truth indices
    for k in ks:
        correct = (ranks[:, :k] == gt).any(dim=1).float().mean().item()
        results[f"M{modalities[0]}->_M{modalities[1]}_recall@{k}"] = correct

    # m2 -> m1 retrieval
    ranks = sims.T.argsort(dim=1, descending=True)  # [N, N]
    for k in ks:
        correct = (ranks[:, :k] == gt).any(dim=1).float().mean().item()
        results[f"M{modalities[1]}->_M{modalities[0]}_recall@{k}"] = correct

    return results