
import re
import torch
import torch.nn.functional as F
import pandas as pd
import random
# from inference import *
import sys
import os
from pathlib import Path

# repo root: pepcraft/tools/Generating/AMPGAN_v3.py -> AMPGANv3/
REPO_ROOT = Path(__file__).resolve().parents[3]


species_dict = {"ecoli": 0,
                "paeruginosa":1,
                "kpneumoniae":2,
                "saureus":3,
                "bsubtilis":4,
                "sepidermidis":5}
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler
import lightning as L

import pandas as pd
import numpy as np

import random
import ast
import copy 
import os
import csv

def identify_tokens(sequence):

    token = []
    token_long = False
    tokening = ""
    for word in sequence:

        if (word != "<" and word !=">") and not token_long:
            token.append(word)
        elif word == "<":
            token_long = True
            tokening += word
        elif token_long and word != ">":
            tokening += word
        elif token_long and word == ">":
            token_long = False
            tokening += word
            token.append(tokening)
            tokening = ""

    return token

def one_hot_encode_sequence(sequence, token_dicts, seq_length):

    one_hot_encoded = np.zeros((len(token_dicts), seq_length))
    tokenized_sequence = identify_tokens(sequence)

    for i, token in enumerate(tokenized_sequence):

        one_hot_encoded[int(token_dicts[token])][i] = 1  


    for i  in range(len(tokenized_sequence), seq_length):

        one_hot_encoded[int(token_dicts['<blank>'])][i] = 1  

    return np.array(one_hot_encoded)

def decode_condition_vectors(condition_labels, conditions):
    species = []
    objects = []
    groups = []
    mic = []
    condition_label_copy = copy.deepcopy(condition_labels)
    for i, condition_label in enumerate(condition_label_copy[0:3]):
        condition_label_copy[i] = {v: k for k, v in condition_label.items()}

    for condition in conditions:


        species.append(condition_label_copy[0][np.where(condition[0:6] == 1)[0][0]])
        objects.append(f'{[condition_label_copy[1][i] for i in np.where(condition[6:11] == 1)[0]]}')
        groups.append(f'{[condition_label_copy[2][i] for i in np.where(condition[11:16] == 1)[0]]}')
        if np.where(condition[16:] == 1)[0][0] == 0:
            mic.append(f'{[0, condition_label_copy[3][np.where(condition[16:] == 1)[0][0]]]}')
        elif np.where(condition[16:] == 1)[0][0] == 9:
            value = condition_label_copy[3][np.where(condition[16:] == 1)[0][0]]
            mic.append(f'{[value,value * 5]}')
        elif np.where(condition[16:] == 1)[0].size != 0:
            mic.append(f'{[condition_label_copy[3][np.where(condition[16:] == 1)[0][0] - 1],condition_label_copy[3][np.where(condition[16:] == 1)[0][0]]]}')
        else:
            mic.append(f'{[9999,10000]}')

    df_dict = {'speices' :species, 
               'objects' :objects, 
               'groups' :groups, 
               'mic' :mic, 
               }
    df = pd.DataFrame(df_dict)
    return df

def find_occurrences(sequence, token):
    occurrences = []
    start = 0

    while True:
        start = sequence.find(token, start)
        if start == -1:
            break
        occurrences.append(start)
        start += len(token)  
    return occurrences


def decode_sequences(tokens, sequences, generate_sample=False):
    tokens_copy = copy.deepcopy(tokens)
    tokens_copy = {int(v): k for k, v in tokens.items()}
    final_sequence = []

    one_hot_encoded = np.zeros((sequences.shape[0],sequences.shape[1], sequences.shape[2]))
    for seq_ind, generated_sequence in enumerate(sequences):

        for i, j in enumerate(np.argmax(generated_sequence, axis = 0)):
            one_hot_encoded[seq_ind][j][i] = 1
    
    for i, sequence in enumerate(one_hot_encoded):
        sequence_list = []
        for row in sequence.T:
            if np.where(row == 1)[0].size == 1:
                sequence_list.append(tokens_copy[np.where(row == 1)[0][0]])
        final_sequence.append("".join(sequence_list))

    if generate_sample:

        cropped_sequence_right = []
        for i, sequence, in enumerate(final_sequence):
            index = find_occurrences(sequence, '<EOS>')
            if len(index) != 0:
                if sequence[index[0] + 5: index[0] + 10] == "<AMD>" and index[0] + 10 < sequences.shape[2]:
                    cropped_sequence_right.append(sequence[:index[0] + 10])
                    # + "<blank>" * (len(sequence) - len(sequence[:index[0] + 10]))
                elif sequence[index[0] + 5: index[0] + 13] == "<cblank>" and index[0] + 13 < sequences.shape[2]:
                    cropped_sequence_right.append(sequence[:index[0] + 13])
                else:
                    cropped_sequence_right.append(sequence[:index[0] + 5])
                    # + "<blank>" * (len(sequence) - len(sequence[:index[0] + 5]))
            else:
                cropped_sequence_right.append(f"{sequence}")


        return cropped_sequence_right
    else:
        return final_sequence

def generate_samples(datasets, batch_size, generator, n_samples, latent_dim, output_file):
    with torch.no_grad():
        generator.eval()
        dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sequences, conditions = [], []

        for batch_idx, samples in enumerate(dataloader):
            (sequence, condition) = samples

            latent_vectors = torch.randn(len(condition), latent_dim)

            condition = condition.to(device).float()
            generated_samples = generator(latent_vectors.to(device), torch.concat((condition[:,0:6], condition[:,16:]), dim = 1).to(device))

            fake_samples = decode_sequences(datasets.tokens_dict, generated_samples.permute(0,2,1).detach().cpu().numpy(),generate_sample=True)
            sequences.append(fake_samples)
            conditions.append(condition.cpu().detach().numpy())
        
        sequences = np.concatenate(sequences)
        conditions = np.concatenate(conditions)

        condition_labels = [datasets.species_dict, datasets.groups_dict, datasets.objects_dict, datasets.bin_edges]

        df = decode_condition_vectors(condition_labels, conditions)

        out_tag = "_concat" 

        df["sequence"] = sequences

        # Drop rows that have the empty sequence
        df = df[df.sequence != ""]
        df = df.iloc[:n_samples]
        df.to_csv(
            output_file.replace(".csv", f"{out_tag}_{str(pd.Timestamp.now())[:10]}.csv")
        )

    

class AMPDatasets(Dataset):
    def __init__(self, data_path = "data/", max_length = 64):

        self.data_path = data_path
        # nterminus and cterminus
        self.max_length = max_length
        
        # labels
        self.target_objects = ['LIPID BILAYER', 'DNA / RNA', 'CYTOPLASMIC PROTEIN', 'MEMBRANE PROTEIN', 'OTHER']
        self.target_groups  = ['GRAM-', 'GRAM+', 'MAMMALIAN CELL', 'FUNGUS', 'OTHER']

        self.species        = ['escherichia coli', 'pseudomonas aeruginosa', 'klebsiella pneumoniae',
                    'staphylococcus aureus', 'bacillus subtilis', 'staphylococcus epidermidis']
        
        self.species_dict = {species_name: i  for i, species_name in enumerate(self.species)}
        self.groups_dict = {groups_name: i for i, groups_name in enumerate(self.target_groups)}
        self.objects_dict = {objects_name: i for i, objects_name in enumerate(self.target_objects)}


        # load datasets
        self.dbaasp_df = pd.read_csv(os.path.join(data_path, "dbaasp.csv"), index_col=0)

        self.dbaasp_df['targetGroups'] = self.dbaasp_df['targetGroups'].apply(ast.literal_eval)
        self.dbaasp_df['targetObjects'] = self.dbaasp_df['targetObjects'].apply(ast.literal_eval)

        # Bins for MIC values
        categories, self.bin_edges = pd.qcut(self.dbaasp_df['MIC'], q=10, labels=False, retbins=True)
        self.dbaasp_df['MIC_category'] = categories


        # get tokens
        self.tokens = np.array(self.dbaasp_df['modified_sequence'].apply(identify_tokens))
        self.tokens = set(list(np.concatenate(self.tokens)) + ['<blank>'])
        self.tokens_dict = {tokens_name: i for i, tokens_name in enumerate(self.tokens)}

        if os.path.exists(os.path.join(self.data_path, "dict.csv")):
            with open(os.path.join(self.data_path, "dict.csv")) as csv_file:
                reader = csv.reader(csv_file)
                temp = {key: int(value) for key, value in reader}

            if len(set(temp.keys()) & self.tokens) == len(self.tokens):
                self.tokens_dict = temp
        else:
            with open(os.path.join(self.data_path, "dict.csv"), "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerows(self.tokens_dict.items())

            print("Created new data/dict.csv")


        # sequences one-hot encoding
        self.sequences = []
        self.conditions = []


        self.dbaasp_df['MIC_norm'] = np.log(self.dbaasp_df['MIC'] + 1e-6)  # Add a small constant to avoid log(0)
        self.log_mean_mic = self.dbaasp_df['MIC_norm'].mean()
        self.log_std_mic = self.dbaasp_df['MIC_norm'].std()

        # Standardize the log-transformed MIC
        self.dbaasp_df['MIC_norm'] = (self.dbaasp_df['MIC_norm'] - self.log_mean_mic) / self.log_std_mic

        self.mic_min = np.min(self.dbaasp_df['MIC_norm'])
        self.mic_max = np.max(self.dbaasp_df['MIC_norm'])

        self.dbaasp_df.sample(frac=1, replace=False, random_state=42).reset_index(drop=True)

        for row in self.dbaasp_df.values:
            encoded_species = np.zeros(6)
            encoded_groups  = np.zeros(5)   
            encoded_objects = np.zeros(5)
            encoded_mic     = np.zeros(10)   

        
            # create binary encoding for conditions (species, group, target and mic)
            encoded_species[self.species_dict[row[0]]] = 1

            for target_group in row[2]:
                encoded_groups[self.groups_dict[target_group]] = 1

            for target_object in row[3]:
                encoded_objects[self.objects_dict[target_object]] = 1

            encoded_mic[row[5]] = 1

            self.sequences.append(one_hot_encode_sequence(row[1], self.tokens_dict, self.max_length))
            #####
            # AMPGAN is not using raw mic rather than one hot encoded MIC
            #####

            self.conditions.append(np.concatenate([encoded_species, encoded_groups, encoded_objects, [row[6]]]))


    def __len__(self):
        return len(self.dbaasp_df)

    def __getitem__(self, idx):
        sample = self.sequences[idx]
        condition = self.conditions[idx]
        label = 1 
        return {'samples':sample, 'conditions':condition,'label':label}
    
class AMPDataModule(L.LightningDataModule):
    def __init__(self, data_path="data/", max_length=68, batch_size=256):
        super().__init__()
        self.data_path = data_path
        self.max_length = max_length
        self.batch_size = batch_size
    def setup(self, stage=None):
        self.full_dataset = AMPDatasets(data_path=self.data_path, max_length=self.max_length)
    def train_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, num_workers=8)

import math

import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[num_tokens, batch_size, embedding_dim]``
        """

        seq_len = x.size(0)
        return self.dropout(x + self.pe[:seq_len])
    
class FiLM(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super(FiLM, self).__init__()

        self.gamma_fc = nn.Linear(condition_dim, feature_dim)
        self.beta_fc = nn.Linear(condition_dim, feature_dim)
        self.init()
    def init(self):
        nn.init.ones_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, x, condition):

        gamma = self.gamma_fc(condition).unsqueeze(2)

        beta = self.beta_fc(condition).unsqueeze(2)

        return gamma * x.unsqueeze(2) + beta

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# from .utils import FiLM, PositionalEncoding

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
        x = x + emb 
    
        x = self.pos_encoder(x.permute(1,0,2)).permute(1,0,2)
        x = self.transformer_encoder(x).permute(0,2,1)
        # print(x.shape)

        # x = self.transformer_encoder(x).permute(0,2,1)
        x = self.final_conv(x).permute(0,2,1)
        

        return x



def AMPGAN_v3(input_data: dict) -> str:

    min_length = input_data.get("min_length", 10)
    max_length = input_data.get("max_length", 100)

    folder_path = input_data.get("folder_path", None)
    csv_path = os.path.join(folder_path, "generated_sequences.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        num_original = len(df)
    else:
        num_original = 0

    species_of_interest = input_data.get("species_of_interest", "ecoli")
    num_samples = input_data.get("num_generations", 4)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BS = 256
    latent_size = 256
    
    dataset = AMPDatasets(max_length=68, data_path=str(REPO_ROOT / "training" / "data"))
    
    # load trained model
    model = Generator(output_shape=(68, len(dataset.tokens)), species_shape=(6,), embed_dim=128).to(DEVICE)
    state_dict = torch.load(REPO_ROOT / "weights" / "Generator_7_200.pth", map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    species, sequences, conditions, lengths, raw = [], [], [], [], []

    mic_values = dataset.dbaasp_df[dataset.dbaasp_df['MIC'] < 32.0]['MIC_norm']
    with torch.no_grad():
        while len(sequences) < num_samples: 
            latent_vectors = torch.randn(BS, latent_size).to(DEVICE)
            
            # species conditioning
            slice1 = torch.zeros(BS, 6).to(DEVICE)
            slice1[:, species_dict[species_of_interest]] = 1

            # MIC conditioning
            slice2 = torch.tensor([random.choice(mic_values.tolist()) for _ in range(BS)], device=DEVICE)
            
            # length conditioning
            seq_length = torch.randint(min_length, max_length, (BS,)).to(DEVICE)
            seq_length = seq_length / 64 # Normalize by max length (64)
            seq_length = seq_length.unsqueeze(1)

            fake_samples = model(latent_vectors, slice1, slice2, seq_length)
            fake_samples = F.gumbel_softmax(fake_samples, tau=0.5, hard=True, dim=2)
            fake_samples = decode_sequences(dataset.tokens_dict, fake_samples.permute(0, 2, 1).detach().cpu().numpy(), generate_sample=False)
            pattern = r'<nblank><SOS>[a-zA-Z]+<EOS><cblank>'

            for sample, length in zip(fake_samples,seq_length):
                if re.findall(pattern, sample):
                    # ONLY L amino acid
                    matches = re.findall(r'<SOS>([a-zA-Z]+)<EOS>', sample)

                    for seq in matches:
                        # amino acid check
                        if seq not in sequences and all(c in 'ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy' for c in seq):
                            sequences.append(seq)
                            raw.append(sample)
                            lengths.append(int(length * 64))
                            species.append(dataset.species[int(slice1[0].argmax())])

    df_dict = {'sequence': sequences[:num_samples],}
    temp_df = pd.DataFrame(df_dict)
    if num_original > 0:
        for col in df.columns:
            if col != "sequence":
                temp_df[col] = "new"
        df = pd.concat([df, temp_df], ignore_index=True)
    else:
        df = temp_df
    df.to_csv(os.path.join(csv_path), index=False)
    # remove the gpu allocation for the model to free up GPU memory
    del model
    torch.cuda.empty_cache()
    
    return (
            f"Successfully generated {len(sequences)} new sequences using AMPGAN-v3 "
            f"(min_length={input_data['min_length']}, max_length={input_data['max_length']}). "
            f"Appended to existing file: {csv_path}. "
            f"(Previous total: {num_original} | New total: {len(df)}). "
            f"Missing filter columns for the new entries were safely filled with NA.\n"
        )
# if __name__ == "__main__":
#     sp = ['ecoli', 'paeruginosa', 'kpneumoniae', 'saureus', 'bsubtilis', 'sepidermidis']
#     for s in sp:
#         input_data = {
#             "species_of_interest": s,
#             "num_generations": 5,
#             "min_length": 15,
#             "max_length": 32,
#             "folder_path": f"/home/raymondlab/Documents/AMP-Agent/output/"
#         }
#         AMPGAN_v3(input_data)