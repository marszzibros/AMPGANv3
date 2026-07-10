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

        if os.path.exists("data/dict.csv"):
            with open("data/dict.csv") as csv_file:
                reader = csv.reader(csv_file)
                temp = {key: int(value) for key, value in reader}

            if len(set(temp.keys()) & self.tokens) == len(self.tokens):
                self.tokens_dict = temp
        else:
            with open("data/dict.csv", "w", newline="") as csv_file:
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

