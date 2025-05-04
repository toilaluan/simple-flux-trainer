import torch.nn as nn

class LatentTextEncoder(nn.Module):
    def __init__(self, text_encoder, num_hidden_layers=8, num_attention_heads=8):
        self.text_encoder = text_encoder
        self.config = self.text_encoder.config
        self.hidden_size = self.config.hidden_size
