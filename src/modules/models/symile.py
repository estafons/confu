import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from symile import Symile, MIPSimilarity

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


class SymileBaselineCLIPModule(pl.LightningModule):
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


        logit_scale_init = 2.5 #-0.3 # np.log(1 / 0.07)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

        self.lr = lr
        self.weight_decay = weight_decay

        self.symile_loss = Symile()

    def forward(self, image_inputs, audio_inputs, text_inputs):
        # encode
      #  print(f"Image inputs shape: {image_inputs.shape}, Audio inputs shape: {audio_inputs.shape}, Text inputs shape: {text_inputs.shape}")
        img_emb, _ = self.modality1_encoder(image_inputs)         # [B, D_img]
        aud_emb, _ = self.modality2_encoder(audio_inputs)         # [B, D_aud]
        txt_emb, _ = self.modality3_encoder(text_inputs)  # [B, D_txt]

        img_emb = self.mod1_proj(img_emb)
        aud_emb = self.mod2_proj(aud_emb)
        txt_emb = self.mod3_proj(txt_emb)
       # print(self.logit_scale.requires_grad)
        mod1_emb = F.normalize(img_emb, dim=-1)  # [B, embed_dim]
        mod2_emb = F.normalize(aud_emb, dim=-1)  # [B, embed_dim]
        mod3_emb = F.normalize(txt_emb, dim=-1)

        # project to common space
        # ia_proj = F.normalize(self.ia_proj(ia_emb), dim=-1)    # [B, embed_dim]
        # txt_proj = F.normalize(self.txt_proj(txt_emb), dim=-1) # [B, embed_dim]

        return mod1_emb, mod2_emb, mod3_emb

    def video_forward(self, image_inputs, audio_inputs, text_inputs):

        mod1_embs = []


        for image in image_inputs:
            
            image = image.to(self.device)
            mod1_emb, mod2_emb, mod3_emb = self.forward(image, audio_inputs, text_inputs)
            mod1_embs.append(mod1_emb)


        mod1_emb = torch.mean(torch.stack(mod1_embs, dim=0), dim=0)
        
        mod1_emb = F.normalize(mod1_emb, dim=-1)  # [B, D]
        return mod1_emb, mod2_emb, mod3_emb



    def training_step(self, batch, batch_idx):
        images, audios, texts, *rest = batch
        
        mod1_emb, mod2_emb, mod3_emb = self(images, audios, texts)
        logit_scale_exp =  self.logit_scale.exp()
        loss = self.symile_loss([mod1_emb, mod2_emb, mod3_emb], logit_scale_exp)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("logit_scale_exp", logit_scale_exp, on_step=True, on_epoch=True, prog_bar=True)
      #  print("requires_grad:", self.logit_scale.requires_grad)
     #   print("grad:", self.logit_scale.grad)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, audios, texts, *rest = batch

        mod1_emb, mod2_emb, mod3_emb = self(images, audios, texts)
        logit_scale_exp =  self.logit_scale.exp()
        loss = self.symile_loss([mod1_emb, mod2_emb, mod3_emb], logit_scale_exp)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    

    def configure_optimizers(self):
        params = [
            {'params': self.modality1_encoder.parameters()},
            {'params': self.modality2_encoder.parameters()},
            {'params': self.modality3_encoder.parameters()},
            {'params': [self.logit_scale], 'weight_decay': 0.0, 'lr':0.1}
        ]
        # params = list(self.modality1_encoder.parameters()) + \
        #          list(self.modality2_encoder.parameters()) + \
        #          list(self.modality3_encoder.parameters()) + \
        #         [self.logit_scale]
        # Use AdamW optimizer
        
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
        mip_similarity = MIPSimilarity()
        similarity_scores = mip_similarity(emb1, [emb2, emb3])
        logit_scale_exp = self.logit_scale.exp()
        logit_scale_exp_copy = logit_scale_exp.detach().clone()
        logit_scale_exp_copy = logit_scale_exp_copy.cpu().item()
        similarity_scores = logit_scale_exp_copy * similarity_scores

        return recall_from_sims(similarity_scores, ks=ks, modalities=modalities)


# def compute_recall_at_k(similarity_scores, k):
#     """
#     Compute Recall@K for the given similarity scores.
#     Args:
#         similarity_scores: Tensor of shape (num_samples, num_samples)
#         k: int, the 'K' in Recall@K
#     Returns:
#         recall: float, the Recall@K value
#     """
#     num_samples = similarity_scores.size(0)

#     # get top-k indices
#     _, indices = similarity_scores.topk(k, dim=1, largest=True, sorted=True)

#     # ground truth matches (sample i should match with index i)
#     gt = torch.arange(num_samples, device=similarity_scores.device).unsqueeze(1)

#     # check if ground truth is within top-k
#     correct = (indices == gt).any(dim=1).float().mean().item()
#     return correct


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