import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .utils import FiLM, PositionalEncoding

class Generator(nn.Module):

    def __init__(self,
        output_shape=(68, 47),
        latent_shape=(256,),
        species_shape=(6,),
        embed_dim=32):
        super().__init__()

        self.latent_shape = latent_shape
        self.output_shape = output_shape
        self.species_shape = species_shape
        self.embed_dim = embed_dim

        # input : Latent Space (256) - 6 Species, Length information (1)
        self.length_embedding = nn.Linear(1, self.embed_dim)


        self.mic_embedding = nn.Linear(1, self.embed_dim)

        self.species_embedding = nn.Embedding(num_embeddings=self.species_shape[0], embedding_dim=self.embed_dim)
        self.embedding_linear = nn.Sequential(nn.Linear(3 * self.embed_dim, 128),
                                              nn.SiLU())

        # FiLM layer
        # https://arxiv.org/abs/1709.07871
        self.film = FiLM(latent_shape[0], 3 * self.embed_dim)

        self.upsampling1 = nn.Sequential(nn.ConvTranspose1d(self.latent_shape[0], 512, kernel_size=4, stride = 1, padding = 0, bias=False),
                                        nn.BatchNorm1d(512),
                                        nn.SiLU(),
                                        nn.ConvTranspose1d(512, 256, kernel_size=4, stride = 2, padding = 1, bias=False),
                                        nn.BatchNorm1d(256),
                                        nn.SiLU(),
                                        nn.ConvTranspose1d(256,128, kernel_size=4, stride = 2, padding = 1, bias=False),
                                        nn.BatchNorm1d(128),
                                        nn.SiLU(),
                                        nn.ConvTranspose1d(128, 128, kernel_size=4, stride=4, padding=0, bias=False),
                                        nn.BatchNorm1d(128),
                                        nn.SiLU())
                                        #     
        self.pos_encoder = PositionalEncoding(d_model=128, dropout=0.2, max_len = 1000)
        encoder_layers = TransformerEncoderLayer(128, 4, 512, dropout=0.2, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 4)

        # final conv layers
        self.final_conv = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(in_channels=64, out_channels=self.output_shape[1], kernel_size=1),

        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        # self.activation_prediction = nn.Softmax(dim=2)


    def forward(self, z, species, mic, length):

        # z (latent space)   : (BS, 256)
        # species : (BS, 6)
        # mic : float
        # length : int

        BS, LS = z.shape

        mic_emb = self.mic_embedding(mic.unsqueeze(1))
        species_idx = torch.argmax(species, dim=-1)
        speices_emb = self.species_embedding(species_idx)
        length_emb = self.length_embedding(length)
        embeddings = torch.cat([mic_emb, speices_emb,length_emb], dim=1)
        alpha = torch.sigmoid(self.alpha)

        z = alpha * z + self.film(z, embeddings).flatten(1)

        
        x = self.upsampling1(z.view(BS, LS, 1))
        x = F.interpolate(x, size=68, mode="linear")
        # BS 128 8
        

        x = x.permute(0,2,1)
        
        emb = self.embedding_linear(embeddings).unsqueeze(1)  # [B,1,D]
        x = torch.cat([emb, x], dim=1)  # prepend
    
        x = self.pos_encoder(x.permute(1,0,2)).permute(1,0,2)
        x = self.transformer_encoder(x)[:,1:,:].permute(0,2,1)
        # print(x.shape)

        # x = self.transformer_encoder(x).permute(0,2,1)
        x = self.final_conv(x).permute(0,2,1)
        

        return x
