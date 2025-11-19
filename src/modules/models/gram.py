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

class GramBaselineCLIPModule(pl.LightningModule):
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


    def volume_computation(self, anchor, *inputs):
        """
        General function to compute volume for contrastive learning loss functions.
        Compute the volume metric for each vector in anchor batch and all the other modalities listed in *inputs.

        Args:
        - anchor (torch.Tensor): Tensor of shape (batch_size1, dim)
        - *inputs (torch.Tensor): Variable number of tensors of shape (batch_size2, dim)

        Returns:
        - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
        """
        batch_size1 = anchor.shape[0]
        batch_size2 = inputs[0].shape[0]

        # Compute pairwise dot products for language with itself
        aa = torch.einsum('bi,bi->b', anchor, anchor).unsqueeze(1).expand(-1, batch_size2)

        # Compute pairwise dot products for language with each input
        l_inputs = [anchor @ input.T for input in inputs]

        # Compute pairwise dot products for each input with themselves and with each other
        input_dot_products = []
        for i, input1 in enumerate(inputs):
            row = []
            for j, input2 in enumerate(inputs):
                dot_product = torch.einsum('bi,bi->b', input1, input2).unsqueeze(0).expand(batch_size1, -1)
                row.append(dot_product)
            input_dot_products.append(row)

        # Stack the results to form the Gram matrix for each pair
        G = torch.stack([
            torch.stack([aa] + l_inputs, dim=-1),
            *[torch.stack([l_inputs[i]] + input_dot_products[i], dim=-1) for i in range(len(inputs))]
        ], dim=-2)

        # Compute the determinant for each Gram matrix
        gram_det = torch.det(G.float())

        # Compute the square root of the absolute value of the determinants
        res = torch.sqrt(torch.abs(gram_det))
        return res
    # def area_computation(self, language, video, audio):
    #     """
    #     Computes pairwise similarity matrix between the combined embeddings.
    #     """
    #     # Example: combine modalities (could be a learned function)
    #     # Here we use dot product between (language, video) and audio.
    #     # You can adapt this to your desired definition of "area".
    #     area = (language @ video.T + language @ audio.T + video @ audio.T) / 3
    #     return area

    def compute_volume_loss(self, language, video, audio):
        
        bs = language.size(0)
        contrastive_temp = 0.07

        # Compute area and areaT
        volume = self.volume_computation(language,video,audio)
        volume = volume / contrastive_temp


        volumeT = self.volume_computation(language,video,audio).T
        volumeT = volumeT / contrastive_temp

        targets = torch.arange(bs, dtype=torch.long, device=self.device)

        loss = (
                F.cross_entropy(-volume, targets, label_smoothing=0.1) #d2a
                + F.cross_entropy(-volumeT, targets, label_smoothing=0.1) #a2d
        ) / 2



        return loss


        

    def training_step(self, batch, batch_idx):
        images, audios, texts, *rest = batch

        mod1_emb, mod2_emb, mod3_emb = self(images, audios, texts)
        loss = self.compute_volume_loss(mod1_emb, mod2_emb, mod3_emb)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, audios, texts, *rest = batch

        mod1_emb, mod2_emb, mod3_emb = self(images, audios, texts)
        loss1_23 = self.compute_volume_loss(mod1_emb, mod2_emb, mod3_emb)
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
        
        similarity_scores = -self.volume_computation(emb1, emb2, emb3)
        
        return recall_from_sims(similarity_scores, ks=ks, modalities=modalities)


    def recall_at_k_2(self, emb1, emb2, ks=[1, 5, 10], modalities=[1, 2]):
        
        similarity_scores = -self.volume_computation(emb1, emb2)
        
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