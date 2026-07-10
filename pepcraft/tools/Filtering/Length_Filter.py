import os
import pandas as pd

def Length_Filter(input_data: dict) -> str:
    folder_path = input_data.get("folder_path", None)
    if folder_path:
        csv_path = os.path.join(folder_path, "generated_sequences.csv")
    else:
        raise ValueError("folder_path is required in input_data")
    
    
    min_length = int(input_data.get("min_length", 10))
    max_length = int(input_data.get("max_length", 32))
    df = pd.read_csv(csv_path)

    length = []
    for sequence in df["sequence"]:

        length.append(1 if (len(sequence) >= min_length and len(sequence) <= max_length) else 0)

    df["length_filter"] = length
    df.to_csv(csv_path, index=False)
    reported = [col for col in df.columns if "filter" in col]
    reported_done = []
    for col in reported:
        if col in df.columns:
            check = df[col].apply(lambda x: True if (x is not None) and (x != "") and (x!="new") else False)
        if check.sum() == len(check):
            reported_done.append(col)
    filter_text = f"Completed Filtering : {', '.join(reported_done)}"

    report = [f"The length filter was applied with a minimum threshold of {min_length} and a maximum threshold of {max_length}.",
              f"The filtered sequences are saved as a new column 'length_filter' in the original CSV file. ",
              f"A value of 1 indicates the sequence passed the filter, while 0 indicates it did not.", filter_text]
    return "\n".join(report)