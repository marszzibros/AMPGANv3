import os
import pandas as pd

from Bio.SeqUtils.ProtParam import ProteinAnalysis

def Lamino_Filter(input_data: dict) -> str:
    folder_path = input_data.get("folder_path", None)
    if folder_path:
        csv_path = os.path.join(folder_path, "generated_sequences.csv")
    else:
        raise ValueError("folder_path is required in input_data")
    df = pd.read_csv(csv_path)

    lamino = []
    for sequence in df["sequence"]:
        # check if the sequence only includes uppercase letters
        if sequence.isupper():
            lamino.append(1)
        else:
            lamino.append(0)    

    df["lamino_filter"] = lamino
    df.to_csv(csv_path, index=False)
    reported = [col for col in df.columns if "filter" in col]
    reported_done = []
    for col in reported:
        if col in df.columns:
            check = df[col].apply(lambda x: True if (x is not None) and (x != "") and (x!="new") else False)
        if check.sum() == len(check):
            reported_done.append(col)
    filter_text = f"Completed Filtering : {', '.join(reported_done)}"

    report = [f"The filtered sequences are saved as a new column 'lamino_filter' in the original CSV file. ",
              f"A value of 1 indicates the sequence passed the filter, while 0 indicates it did not.",filter_text]
    return "\n".join(report)