import torch
import torch.nn as nn



class Transformer(nn.Module):
    """Extends nn.Transformer."""
    
    def __init__(self, n_features, dim, num_layers=5):
        """Initialize Transformer object.

        Args:
            n_features (int): Number of features in the input.
            dim (int): Dimension which to embed upon / Hidden dimension size.
        """
        super().__init__()
        self.embed_dim = dim
        self.output_dim = dim
        self.input_norm = nn.LayerNorm(n_features)
        self.conv = nn.Conv1d(n_features, self.embed_dim,
                              kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=5)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        """Apply Transformer to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        if type(x) is list:
            x = x[0]
        x = self.input_norm(x)  # Normalize input: (batch, n_features, seq_len)
        x = self.conv(x.permute([0, 2, 1]))
        x = x.permute([2, 0, 1])
        x_l = self.transformer(x)[-1]
        seq = x.permute([1, 0, 2])
        return x_l, seq
    


import torch
import torch.nn as nn

class FusionTransformer(nn.Module):
    """Extends nn.Transformer with CLS token."""
    
    def __init__(self, n_features, dim, num_layers=5, nhead=5):
        """Initialize Transformer object.

        Args:
            n_features (int): Number of features in the input.
            dim (int): Dimension which to embed upon / Hidden dimension size.
        """
        super().__init__()
        self.embed_dim = dim
        self.output_dim = dim

        self.input_norm = nn.LayerNorm(n_features)
        self.conv = nn.Conv1d(n_features, self.embed_dim,
                              kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

    def forward(self, x):
        """Apply Transformer to Input.

        Args:
            x (torch.Tensor): Layer Input (batch, n_features, seq_len)

        Returns:
            x_cls (torch.Tensor): CLS embedding (batch, dim)
            seq (torch.Tensor): Sequence embeddings (batch, seq_len, dim)
        """
        if isinstance(x, list):
            x = x[0]

        # Keep your normalization as-is
        x = self.input_norm(x)                          # (batch, n_features, seq_len)

        x = self.conv(x.permute(0, 2, 1))               # (batch, embed_dim, seq_len)
        x = x.permute(2, 0, 1)                          # (seq_len, batch, embed_dim)

        # Add CLS token
        batch_size = x.size(1)
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)  # (1, batch, embed_dim)
        x = torch.cat([cls_tokens, x], dim=0)           # (seq_len+1, batch, embed_dim)

        # Transformer
        x = self.transformer(x)                         # (seq_len+1, batch, embed_dim)

        # CLS token is the first position
        x_cls = x[0]                                    # (batch, embed_dim)

        # Sequence embeddings (exclude CLS)
        seq = x[1:].permute(1, 0, 2)                    # (batch, seq_len, embed_dim)

        return x_cls, seq

