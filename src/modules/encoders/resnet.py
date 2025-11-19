import torch.nn as nn
from torchvision.models import resnet18
from transformers import AutoModel, AutoConfig, AutoTokenizer

def create_image_encoder(embed_dim=256, in_channels=1, backbone='resnet18'):
    """
    Create a ResNet18 image encoder returning (embedding, None)
    
    Args:
        embed_dim (int): Output embedding dimension.
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB).
    
    Returns:
        nn.Module: ResNet18 encoder wrapped to return a tuple (embedding, None)
    """
    if backbone == 'resnet18':
        base_model = resnet18(weights=None)  # no pretrained weights
    elif backbone == 'resnet50':
        from torchvision.models import resnet50
        from torchvision.models import ResNet50_Weights
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Modify first conv layer if needed (e.g., for grayscale images)
    if in_channels != 3:
        base_model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=base_model.conv1.out_channels,
            kernel_size=base_model.conv1.kernel_size,
            stride=base_model.conv1.stride,
            padding=base_model.conv1.padding,
            bias=base_model.conv1.bias is not None
        )

    # Replace the classifier with a projection to embed_dim
    in_features = base_model.fc.in_features
    base_model.fc = nn.Linear(in_features, embed_dim)

    # Wrap in a module to return two outputs
    class ImageEncoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.output_dim = embed_dim

        def forward(self, x):
            emb = self.model(x)
            return emb, None  # second output can be None

    return ImageEncoder(base_model)



def create_audio_encoder(embed_dim=256, in_channels=1, backbone='resnet18'):
    """
    Audio encoder returning a tuple: (embedding, optional second output)
    """
    if backbone == 'resnet18':

        base_model = resnet18(weights=None)
    elif backbone == 'resnet50':
        from torchvision.models import resnet50
        from torchvision.models import ResNet50_Weights
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Modify first conv layer for single-channel spectrograms
    base_model.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=base_model.conv1.out_channels,
        kernel_size=base_model.conv1.kernel_size,
        stride=base_model.conv1.stride,
        padding=base_model.conv1.padding,
        bias=False
    )

    # Replace classifier with projection
    in_features = base_model.fc.in_features
    base_model.fc = nn.Linear(in_features, embed_dim)

    # Wrap in a module to return two outputs
    class AudioEncoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.output_dim = embed_dim

        def forward(self, x):
            emb = self.model(x)
            return emb, None  # second output can be None

    return AudioEncoder(base_model)


def create_text_encoder(embed_dim=256, device='cpu'):
    """
    MiniLM text encoder returning (embedding, None). Handles tokenization internally.
    """
    # Load config and model from scratch (random weights)
    config = AutoConfig.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    config.output_hidden_states = True
    base_model = AutoModel.from_config(config)
    
    # Projection layer to desired embed_dim
    projection = nn.Linear(config.hidden_size, embed_dim)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    class TextEncoder(nn.Module):
        def __init__(self, model, proj, tokenizer):
            super().__init__()
            self.model = model
            self.projection = proj
            self.tokenizer = tokenizer
            self.output_dim = embed_dim

        def forward(self, texts):
            """
            texts: list of strings
            """
            # Tokenize batch of texts
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids'].to(self.projection.weight.device)
            attention_mask = encoded['attention_mask'].to(self.projection.weight.device)
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # CLS token
            return self.projection(cls_emb), None  # (batch, embed_dim), None

    return TextEncoder(base_model, projection, tokenizer)
