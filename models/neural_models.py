import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class FeedforwardRegressor(nn.Module):
    def __init__(self, input_count=2, output_count=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_count, 200), nn.ReLU(),
            nn.Linear(200, 300), nn.ReLU(),
            nn.Linear(300, 500), nn.ReLU(),
            nn.Linear(500, 500), nn.ReLU(),
            nn.Linear(500, 300), nn.ReLU(),
            nn.Linear(300, 200), nn.ReLU(),
            nn.Linear(200, output_count)
        )

    def forward(self, x):
        return self.network(x)


class TransformerRegressor(nn.Module):
    def __init__(self, input_count=2, output_count=1,
                 dim_model=100, num_heads=2,
                 num_encoder_layers=5, dim_hidden=500, dropout_p=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_count, dim_model)
        encoder_layer = TransformerEncoderLayer(
            dim_model, num_heads, dim_hidden, dropout_p, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.out = nn.Linear(dim_model, output_count)

    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        transformer_out = self.transformer(src, src_mask)
        return self.out(transformer_out)
