import os
import pandas as pd
import json
import datetime
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.MMCIFParser import MMCIFParser

def convert_cif_to_pdb(cif_path):
    # 1. Convert CIF to PDB
    parser_cif = MMCIFParser(QUIET=True)
    structure = parser_cif.get_structure("peptide", cif_path)

    pdb_path = cif_path.replace(".cif", ".pdb")
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_path)

    with open(pdb_path, 'r') as f:
        lines = f.readlines()

    dummy_cryst = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1          \n"

    if not lines[0].startswith("CRYST1"):
        with open(pdb_path, 'w') as f:
            f.write(dummy_cryst)
            f.writelines(lines)
    return pdb_path

def Predict_Structure(input_data: dict) -> str:
    # predict structure using simple-fold, for demonstration.

    folder_path = input_data.get("folder_path", None)
    structure_path = input_data.get("structure_path", None)

    csv_path = os.path.join(folder_path, "generated_sequences.csv")

    df = pd.read_csv(csv_path)
    # remove structure_filter columns
    df = df.drop(columns=[col for col in df.columns if "structure_filter" in col])
    keys = []
    for key in df.keys():
        if "filter" in key:
            keys.append(key)

    os.system("mkdir "+os.path.join(folder_path, "fasta_input"))

    if structure_path is None or not os.path.exists(structure_path):
        structure_path = os.path.join(folder_path, f"structure_output")
        os.system(f"mkdir {structure_path}")
    sequences = []
    pdb_path = []
    fasta_path = []
    for index, row in df.iterrows():
        if all(row[key] == 1 for key in keys) and ("predicted_structure_path" not in df.columns or ".pdb" not in str(row["predicted_structure_path"])):
            seq = row["sequence"]
            seq = seq.upper()
            sequences.append(seq)
            now_time = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

            input_fasta_path = os.path.join(folder_path, "fasta_input", f"generated_{now_time}.fasta")
            
            with open(input_fasta_path, "w") as f:
                f.write(f">generated_{now_time}\n")
                f.write(seq + "\n") 
            os.system(f"rm -r {structure_path}/records")
            os.system(f"rm -r {structure_path}/structures")
            os.system(f"rm -r {structure_path}/manifest.json")

            script = f"simplefold \
                      --simplefold_model simplefold_360M \
                      --num_steps 500 --tau 0.01 \
                      --nsample_per_protein 1 \
                      --fasta_path {input_fasta_path} \
                        --output_dir {structure_path} \
                        --backend torch"
            os.system(script)
            
            fasta_path.append(input_fasta_path)
            
            pdb_path.append(convert_cif_to_pdb(os.path.join(structure_path, f"predictions_simplefold_360M", f"generated_{now_time}_sampled_0.cif")))
        elif "predicted_structure_path" in df.columns and ".pdb" in str(row["predicted_structure_path"]):
            pdb_path.append(str(row["predicted_structure_path"]))
            fasta_path.append(row["fasta_path"])
        else:
            pdb_path.append("")
            fasta_path.append("")

    df['predicted_structure_path'] = pdb_path
    df['fasta_path'] = fasta_path
    df.to_csv(csv_path, index=False)

    return f"Structure prediction completed for {len(sequences)} sequences. The predicted structures are saved in {structure_path}."        
# if __name__ == "__main__":

#     result = Predict_Structure({"folder_path": "/home/raymondlab/Documents/AMP-Agent/output/"})
#     print(result)