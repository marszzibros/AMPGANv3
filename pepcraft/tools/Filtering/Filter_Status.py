import os
import pandas as pd

def Filter_Status(input_data: dict) -> str:
    folder_path = input_data.get("folder_path", None)
    if folder_path:
        csv_path = os.path.join(folder_path, "generated_sequences.csv")
    else:
        raise ValueError("folder_path is required in input_data")

    df = pd.read_csv(csv_path)
    filter_columns = [col for col in df.columns if "filter" in col]
    
    passed_sequences = df[df[filter_columns].all(axis=1)]
    passed_count = len(passed_sequences)
    total_count = len(df)

    # check if any entry's value is "new"
    is_there_new = (df[filter_columns] == "new").any().any()
    report = ""
    if is_there_new:
        # report the column.split("_")[0] for each column that has "new" value
        new_columns = df[filter_columns].columns[(df[filter_columns] == "new").any()]
        new_filters = [col.split("_")[0] for col in new_columns]
        report = f"The following filters have not been applied yet: {', '.join(new_filters)}. Please check whether they depend on tools."
    report += f"Among the {total_count} generated sequences, {passed_count} sequences passed all filters ({','.join(filter_columns)})."
    # # for each filter column, report how many sequences passed that specific filter
    # for col in filter_columns:
    #     col_passed_count = df[df[col] == 1].shape[0]
    #     report += f"\n{col}: {col_passed_count} sequences passed this filter."
    return report
