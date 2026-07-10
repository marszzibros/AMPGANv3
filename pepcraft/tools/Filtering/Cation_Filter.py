import os
import pandas as pd

from Bio.SeqUtils.ProtParam import ProteinAnalysis

def Cation_Filter(input_data: dict) -> str:
    folder_path = input_data.get("folder_path", None)
    if folder_path:
        csv_path = os.path.join(folder_path, "generated_sequences.csv")
    else:
        raise ValueError("folder_path is required in input_data")
    min_cationicity = float(input_data.get("min_cationicity", 0.0))
    max_cationicity = float(input_data.get("max_cationicity", 1.0))
    df = pd.read_csv(csv_path)

    cationicity = []
    for sequence in df["sequence"]:
        sequence = sequence.upper()
        analysed_seq = ProteinAnalysis(sequence)
        net_charge = analysed_seq.charge_at_pH(7.4)
        cationicity.append(1 if (net_charge >= min_cationicity and net_charge <= max_cationicity) else 0)

    df["cationicity_filter"] = cationicity
    df.to_csv(csv_path, index=False)

    reported = [col for col in df.columns if "filter" in col]
    reported_done = []
    for col in reported:
        if col in df.columns:
            check = df[col].apply(lambda x: True if (x is not None) and (x != "") and (x!="new") else False)
        if check.sum() == len(check):
            reported_done.append(col)
    filter_text = f"Completed Filtering : {', '.join(reported_done)}"

    report = [f"The cationicity filter was applied with a minimum threshold of {min_cationicity} and a maximum threshold of {max_cationicity}.",
              f"The filtered sequences are saved as a new column 'cationicity_filter' in the original CSV file. ",
              f"A value of 1 indicates the sequence passed the filter, while 0 indicates it did not.", filter_text]
    return "\n".join(report)