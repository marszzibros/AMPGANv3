import math 
 
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .utils import PositionalEncoding

class Discriminator(nn.Module):
    def __init__ (self, 
                  classes: int,
                  n_tokens: int,
                  d_model: int, 
                  nhead: int, 
                  d_hid:int,
                  nlayers: int,
                  seq_len: int,
                  model_type: str,
                  n_species: int = 6,
                  n_conditions: int = 0,
                  dropout: float = 0.5):
        super(Discriminator, self).__init__()

        self.classes = classes
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.seq_len = seq_len
        self.model_type = model_type
        self.n_conditions = n_conditions
        self.n_species = n_species
        self.dropout = dropout

        # Embedding
        self.species_embedding = nn.Embedding(num_embeddings=self.n_species, embedding_dim=self.d_model)

        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=self.n_tokens , out_channels=self.d_model , kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(self.d_model),
            nn.SiLU(),
        )


        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=self.d_model, dropout=self.dropout, max_len = 512)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model ))
        # Encoder
        encoder_layers = TransformerEncoderLayer(d_model=self.d_model , 
                                                 nhead=self.nhead,
                                                 dim_feedforward=self.d_hid, 
                                                 dropout=self.dropout,
                                                 batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)

        # Final Layer
        self.final_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(self.d_model, self.classes)
        )
    def forward(self, x: Tensor, species: Tensor = None) -> Tensor:
        """
        Arguments:
            x: Tensor, [batch_size, n_tokens, seq_len]
            conditions: Tensor, [n_conditions]


        Returns:
            output Tensor: [batch_size, n_classes]
        """
        # x   : (BS, n_tokens, seq_len)
        BS, NT, SL  = x.shape

        x = self.conv_layer(x).permute(0,2,1)

        # gan_cla - None
        # mic_cla - Species (6)
        if self.model_type == "mic_cla":
            species_idx = torch.argmax(species, dim=-1) 
            speices_token= self.species_embedding(species_idx).unsqueeze(1).to(x.device)
            x = torch.cat([ x, speices_token], axis=1)
        BS, C, E = x.shape

        # Add CLS Token
        CLS_token = self.cls_token.repeat(BS, 1, 1)
        x = torch.cat([CLS_token, x], axis=1)

        x = self.pos_encoder(x.permute(1,0,2)).permute(1,0,2)
        
        # Transformer encoder
        # x   : (1 + n_conditions + n_tokens, BS, d_model)
        encoder_output = self.transformer_encoder(x)

        # Final Layer
        output = self.final_layer(encoder_output[:,0,:])
        # x   : (BS, n_classes)

        return output.squeeze(-1)
        