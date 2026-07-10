import os
import pandas as pd

from Bio.SeqUtils.ProtParam import ProteinAnalysis

def Damino_Filter(input_data: dict) -> str:
    folder_path = input_data.get("folder_path", None)
    if folder_path:
        csv_path = os.path.join(folder_path, "generated_sequences.csv")
    else:
        raise ValueError("folder_path is required in input_data")
    df = pd.read_csv(csv_path)

    damino = []
    for sequence in df["sequence"]:
        # check if there is a small letter in the sequence
        if any(c.islower() for c in sequence):
            damino.append(1)
        else:
            damino.append(0)
    df["damino_filter"] = damino
    df.to_csv(csv_path, index=False)
    reported = [col for col in df.columns if "filter" in col]
    reported_done = []
    for col in reported:
        if col in df.columns:
            check = df[col].apply(lambda x: True if (x is not None) and (x != "") and (x!="new") else False)
        if check.sum() == len(check):
            reported_done.append(col)
    filter_text = f"Completed Filtering : {', '.join(reported_done)}"

    report = [f"The filtered sequences are saved as a new column 'damino_filter' in the original CSV file. ",
              f"A value of 1 indicates the sequence passed the filter, while 0 indicates it did not.", filter_text]
    return "\n".join(report)