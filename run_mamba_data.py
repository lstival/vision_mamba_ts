from src.s_mamba.data_provider import data_loader

import torch
import torch.nn as nn

class TimeSeriesTokenizer(nn.Module):
    """
    Tokenizes a time series into patch embeddings using a linear layer.
    """
    def __init__(
        self, 
        sequence_length: int = 384,
        patch_size: int = 16,
        embedding_dim: int = 256,
        num_features: int = 1,
        add_positional_encoding: bool = True
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_features = num_features
        self.add_positional_encoding = add_positional_encoding

        assert sequence_length % patch_size == 0, "Sequence length must be divisible by patch size"
        self.num_tokens = sequence_length // patch_size

        self.patch_projection = nn.Linear(patch_size * num_features, embedding_dim)
        if add_positional_encoding:
            self.positional_encoding = nn.Parameter(
                torch.randn(1, self.num_tokens, embedding_dim) * 0.02
            )

    def create_patches(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, sequence_length, num_features)
        batch_size, seq_len, num_features = x.shape
        patches = x.view(
            batch_size, 
            seq_len // self.patch_size, 
            self.patch_size * num_features
        )
        return patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, sequence_length, num_features)
        patches = self.create_patches(x)
        tokens = self.patch_projection(patches)
        if self.add_positional_encoding:
            tokens = tokens + self.positional_encoding
        return tokens
    

if __name__ == '__main__':
    root_data = r"C:\WUR\ICML\data\S-Mamba_datasets\traffic"
    dataset = data_loader.Dataset_Pred(root_path=root_data,data_path="traffic.csv")

    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(dataset))
    batch_x.resize([1,batch_x.shape[0], batch_x.shape[1]])# increase on dimention first
    x = torch.tensor(batch_x).float()

    tokenizer = TimeSeriesTokenizer(sequence_length=384, patch_size=16, embedding_dim=256, num_features=1)
    tokens = tokenizer(x)