
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Configurable MLP with N layers (default=2)."""

    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, 
                 output_each_layer=False, num_layers=2):
        """
        Args:
            indim (int): Input dimension
            hiddim (int): Hidden layer dimension
            outdim (int): Output layer dimension
            dropout (bool): Whether to apply dropout
            dropoutp (float): Dropout probability
            output_each_layer (bool): Whether to return outputs of each layer
            num_layers (int): Number of layers (>=2)
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropoutp)
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

        # Build layers
        layers = []
        in_features = indim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_features, hiddim))
            in_features = hiddim
        layers.append(nn.Linear(in_features, outdim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        outputs = [0, x] if self.output_each_layer else None

        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ReLU after all except final layer
            if i < len(self.layers) - 1:
                x = F.relu(x)
            else:
                x = self.lklu(x)
            if self.dropout:
                x = self.dropout_layer(x)

            if self.output_each_layer:
                outputs.append(x)

        return outputs if self.output_each_layer else x


# class MLP(torch.nn.Module):
#     """Two layered perceptron."""
    
#     def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, output_each_layer=False, num_layers=2):
#         """Initialize two-layered perceptron.

#         Args:
#             indim (int): Input dimension
#             hiddim (int): Hidden layer dimension
#             outdim (int): Output layer dimension
#             dropout (bool, optional): Whether to apply dropout or not. Defaults to False.
#             dropoutp (float, optional): Dropout probability. Defaults to 0.1.
#             output_each_layer (bool, optional): Whether to return outputs of each layer as a list. Defaults to False.
#         """
#         super(MLP, self).__init__()
#         self.fc = nn.Linear(indim, hiddim)
#         self.fc2 = nn.Linear(hiddim, outdim)
#         self.dropout_layer = torch.nn.Dropout(dropoutp)
#         self.dropout = dropout
#         self.output_each_layer = output_each_layer
#         self.lklu = nn.LeakyReLU(0.2)

#     def forward(self, x):
#         """Apply MLP to Input.

#         Args:
#             x (torch.Tensor): Layer Input

#         Returns:
#             torch.Tensor: Layer Output
#         """
#         output = F.relu(self.fc(x))
#         if self.dropout:
#             output = self.dropout_layer(output)
#         output2 = self.fc2(output)
#         if self.dropout:
#             output2 = self.dropout_layer(output2)
#         if self.output_each_layer:
#             return [0, x, output, self.lklu(output2)]
#         return output2