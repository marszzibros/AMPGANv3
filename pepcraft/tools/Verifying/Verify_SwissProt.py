import os
import subprocess
import pandas as pd
from Bio.Blast import NCBIXML
from Bio import Entrez, SeqIO
def get_taxonomy_and_functional_text(accession_id):

    # NCBI requires an email address to use their Entrez servers
    Entrez.email = 'jay-hwasung.jung@uvm.edu'
    
    
    try:
        # Fetch the protein record from the Entrez Protein database in GenBank format
        handle = Entrez.efetch(db="protein", id=accession_id, rettype="gb", retmode="text")
        record = SeqIO.read(handle, "genbank")
        handle.close()
        
        # Extract Taxonomy
        taxonomy = record.annotations.get("taxonomy", [])
        
        # Extract general description text
        description = record.description
        
        products = []
        for feature in record.features:
            
            if "product" in feature.qualifiers:
                products.extend(feature.qualifiers["product"])
                
        # Remove duplicates from lists
        products = list(set(products))
        taxonomy_str = "--- Taxonomy & Functional Info ---"
        taxonomy_str += f"\nTaxonomy: {' -> '.join(taxonomy)}"
        taxonomy_str += f"\nDescription: {description}"
        taxonomy_str += f"\nProducts: {', '.join(products)}"

        
        return taxonomy_str

    except Exception as e:
        return "No data available due to error."
    
def Verify_SwissProt(input_data: dict) -> str:
    folder_path = input_data.get("folder_path", None)
    if folder_path:
        csv_path = os.path.join(folder_path, "generated_sequences.csv")
    else:
        raise ValueError("folder_path is required in input_data")

    df = pd.read_csv(csv_path)
    filter_columns = [col for col in df.columns if "filter" in col]
    swissprot_hits = os.path.join(folder_path, 'swissprot_hits')

    os.system(f"mkdir -p {swissprot_hits}") # Added -p to prevent errors if folder exists
    BLAST_results = ""
    swissprot_report = []

    for index, row in df.iterrows():
        current_row_text = "" # Reset for EACH row

        try:
            # dbaasp_report exists, and row['dbaasp_report' is not empty, skip
            if 'swissprot_report' in df.columns and row.get('swissprot_report') and row['swissprot_report']!= "new":
                swissprot_report.append(row['swissprot_report'])
                continue
            if all(row[key] == 1 for key in filter_columns):
                current_row_text += f"{row['sequence']}\n"
                fasta = row["fasta_path"]
                hit_xml = os.path.join(swissprot_hits, fasta.split("/")[-1].replace(".fasta", ".xml"))
            
                subprocess.run([
                    "blastp", 
                    "-query", fasta, 
                    "-db", "/home/raymondlab/Documents/AMP-Agent/src/tools/Verifying/swissprot/swissprot",
                    "-outfmt", "5",              
                    "-max_target_seqs", "5",
                    "-task", "blastp-short",     # Optimized for peptides < 30 residues      
                    "-num_threads", "8",
                    "-evalue", "10.0",           # Optional: Adjust based on your novelty threshold
                    "-out", hit_xml
                ], check=True)
                
                parsed_results = []
                
                with open(hit_xml, "r") as result_handle:
                    blast_records = NCBIXML.parse(result_handle)
                    
                    for i, blast_record in enumerate(blast_records):
                        query_id = blast_record.query
                        
                        if not blast_record.alignments:
                            print(f"[{i+1}] No matches found.")
                            parsed_results.append({"query": query_id, "error": "No matches"})
                            continue
                        
                        top_alignment = blast_record.alignments[0]
                        top_hsp = top_alignment.hsps[0] 
                        
                        identity_pct = (top_hsp.identities / top_hsp.align_length) * 100

                        hit_info = {
                            "query": query_id,
                            "title": top_alignment.title,
                            "accession": top_alignment.accession,
                            "length": top_alignment.length,
                            "e_value": top_hsp.expect,
                            "identity_pct": round(identity_pct, 2),
                            "alignment_query": top_hsp.query,
                            "alignment_match": top_hsp.match, 
                            "alignment_subject": top_hsp.sbjct
                        }
                        
                        parsed_results.append(hit_info)
                        current_row_text += f"[{i+1}] Top Hit: {hit_info['accession']} | Identity: {hit_info['identity_pct']}% | e_value: {hit_info['e_value']}\n"
                    
                    taxonomy_info = get_taxonomy_and_functional_text(parsed_results[0]['accession']) if parsed_results else 'No hits, so no taxonomy info.'
                    current_row_text += f"\n{taxonomy_info}\n\n"
                    
                    # Add this row's specific text to the DataFrame list
                    swissprot_report.append(current_row_text)

                    # Add this row's specific text to the master string
                    BLAST_results += current_row_text

            else:
                swissprot_report.append("") # Didn't pass filters

        except Exception as e:
            error_msg = f"No hits or error during BLAST search: {e}\n"
            swissprot_report.append(error_msg)
            BLAST_results += error_msg
    if len(swissprot_report) == 0:
        return "No sequences passed all filters."
    
    df['swissprot_report'] = swissprot_report
    df.to_csv(csv_path, index=False)

    reported = [col for col in df.columns if "report" in col]
    reported_done = []
    for col in reported:
        if col in df.columns:
            check = df[col].apply(lambda x: True if (x is not None) and (x != "")  and (x!="new") else False)
        if check.sum() == len(check):
            reported_done.append([col,len(check)])
    BLAST_results += f"Completed Verification :"
    for col in reported_done:
        BLAST_results += f" {col[0]} ({col[1]} sequences completed),"
    df.to_csv(csv_path, index=False)



    return BLAST_results
