import os
import pandas as pd

from Bio.SeqUtils.ProtParam import ProteinAnalysis

def Hydrophobicity_Filter(input_data: dict) -> str:
    folder_path = input_data.get("folder_path", None)
    if folder_path:
        csv_path = os.path.join(folder_path, "generated_sequences.csv")
    else:
        raise ValueError("folder_path is required in input_data")
    min_hydrophobicity = float(input_data.get("min_hydrophobicity", 0.0))
    max_hydrophobicity = float(input_data.get("max_hydrophobicity", 1.0))
    df = pd.read_csv(csv_path)

    hydrophobicity = []
    for sequence in df["sequence"]:
        sequence = sequence.upper()
        analysed_seq = ProteinAnalysis(sequence)
        gravy = analysed_seq.gravy()
        hydrophobicity.append(1 if (gravy >= min_hydrophobicity and gravy <= max_hydrophobicity) else 0)

    df["hydrophobicity_filter"] = hydrophobicity
    df.to_csv(csv_path, index=False)
    reported = [col for col in df.columns if "filter" in col]
    reported_done = []
    for col in reported:
        if col in df.columns:
            check = df[col].apply(lambda x: True if (x is not None) and (x != "") and (x!="new") else False)
        if check.sum() == len(check):
            reported_done.append(col)
    filter_text = f"Completed Filtering : {', '.join(reported_done)}"

    report = [f"The hydrophobicity filter was applied with a minimum threshold of {min_hydrophobicity} and a maximum threshold of {max_hydrophobicity}.",
              f"The filtered sequences are saved as a new column 'hydrophobicity_filter' in the original CSV file. ",
              f"A value of 1 indicates the sequence passed the filter, while 0 indicates it did not.", filter_text]
    return "\n".join(report)