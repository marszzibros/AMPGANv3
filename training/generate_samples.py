
import re
import torch
import torch.nn.functional as F
import pandas as pd
import random
from inference import *
import sys
import os

os.system("mkdir test/")
os.system(f"mkdir test/{sys.argv[1]}")

species_dict = {"ecoli": 0,
                "paeruginosa":1,
                "kpneumoniae":2,
                "saureus":3,
                "bsubtilis":4,
                "sepidermidis":5}

def generate_samples(input_data: dict) -> str:

    species_of_interest = input_data.get("species_of_interest", "ecoli")
    num_samples = input_data.get("num_samples", 4)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BS = 256
    latent_size = 256
    
    dataset = AMPDatasets(max_length=68, data_path="/gpfs1/home/j/j/jjung2/scratch/AMPGANv3/training/data")
    
    # load trained model
    model = Generator(output_shape=(68, len(dataset.tokens)), species_shape=(6,), embed_dim=128).to(DEVICE)
    state_dict = torch.load(f"/gpfs1/home/j/j/jjung2/scratch/AMPGANv3/training/logs/Generator/Generator_{sys.argv[1]}_100.pth", weights_only=True)
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
            seq_length = torch.randint(15, 32, (BS,)).to(DEVICE)
            seq_length = seq_length / 64 # Normalize by max length (64)
            seq_length = seq_length.unsqueeze(1)

            fake_samples = model(latent_vectors, slice1, slice2, seq_length)
            fake_samples = F.gumbel_softmax(fake_samples, tau=0.5, hard=True, dim=2)
            fake_samples = decode_sequences(dataset.tokens_dict, fake_samples.permute(0, 2, 1).detach().cpu().numpy(), generate_sample=False)
            pattern = r'(<C\d+>|<nblank>|<ACT>)<SOS>[a-zA-Z]+<EOS>(<AMD>|<cblank>)'

    #         for sample, length in zip(fake_samples,seq_length):
    #             if re.findall(pattern, sample):
    #                 # ONLY L amino acid
    #                 matches = re.findall(r'<SOS>([A-Z]+)<EOS>', sample)

    #                 for seq in matches:
    #                     # amino acid check
    #                     if seq not in sequences and all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in seq):
    #                         sequences.append(seq)
    #                         raw.append(sample)
    #                         lengths.append(int(length * 64))
    #                         species.append(dataset.species[int(slice1[0].argmax())])

    # df_dict = {'species': species[:num_samples], 'length': lengths[:num_samples], 'raw': raw[:num_samples], 'sequence': sequences[:num_samples],}
    # pd.DataFrame(df_dict).to_csv(f"generated_samples_{s}.csv", index=False)
    
            for sample, length in zip(fake_samples,seq_length):
                sequences.append(sample)
                lengths.append(int(length * 64))
                species.append(dataset.species[int(slice1[0].argmax())])

                # if re.findall(pattern, sample):
                #     matches = re.findall(r'<SOS>([a-zA-Z]+)<EOS>', sample)

                #     for seq in matches:
                #         if seq not in sequences:
                #             sequences.append(seq)
                #             raw.append(sample)
                #             lengths.append(int(length * 64))
                #             species.append(dataset.species[int(slice1[0].argmax())])

    df_dict = {'species': species[:num_samples], 'length': lengths[:num_samples], 'sequence': sequences[:num_samples]} #, 'sequence': sequences[:num_samples],}
    pd.DataFrame(df_dict).to_csv(f"test/{sys.argv[1]}/generated_samples_{s}.csv", index=False)
    
    return "done!"

if __name__ == "__main__":
    sp = ['ecoli', 'paeruginosa', 'kpneumoniae', 'saureus', 'bsubtilis', 'sepidermidis']
    for s in sp:
        input_data = {
            "species_of_interest": s,
            "num_samples": 1000
        }
        generate_samples(input_data)
