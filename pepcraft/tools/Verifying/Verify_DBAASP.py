import os
import subprocess
import requests
import pandas as pd
from Bio.Blast import NCBIXML
import json
import logging

# Setup logging to track errors without crashing the script
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def safe_float(value):
    """Safely convert MIC strings to floats."""
    if value is None:
        return float('inf')
    try:
        clean_val = ''.join(c for c in str(value) if c.isdigit() or c == '.')
        return float(clean_val) if clean_val else float('inf')
    except ValueError:
        return float('inf')

def fetch_dbaasp(url):
    """Generic helper for DBAASP API requests."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except (requests.exceptions.RequestException, ValueError) as e:
        logging.error(f"API Error at {url}: {e}")
        return None

def parse_llm_ready_summary(data):
    """Extracts high-value fields from DBAASP JSON safely."""
    if not isinstance(data, dict):
        return {"error": "Invalid data format"}

    summary = {
        "name": data.get("name") or "Unknown",
        "sequence": data.get("sequence") or "N/A",
        "charge": "Unknown",
        "hydrophobic_moment": "Unknown",
        "efficacy": [],
        "toxicity": [],
        "sources": []
    }

    # Extract Physico-Chemical properties safely
    for prop in data.get("physicoChemicalProperties") or []:
        if not isinstance(prop, dict): continue
        name = prop.get("name")
        if name == "Net Charge":
            summary["charge"] = prop.get("value")
        elif name == "Normalized Hydrophobic Moment":
            summary["hydrophobic_moment"] = prop.get("value")

    # Extract Efficacy (Top 2 MICs)
    effs = []
    for act in data.get("targetActivities") or []:
        if not isinstance(act, dict): continue
        
        mic = act.get("concentration")
        
        # Safe nested gets: falls back to {} if the API returns 'null'
        species_dict = act.get("targetSpecies") or {}
        species = species_dict.get("name")
        
        measure_dict = act.get("activityMeasureGroup") or {}
        measure = str(measure_dict.get("name", "")).upper()
        
        unit_dict = act.get("unit") or {}
        unit = unit_dict.get("name", "")
        
        if species and mic and "MIC" in measure:
            effs.append({
                "pathogen": species,
                "mic": mic,
                "unit": unit
            })
    
    effs.sort(key=lambda x: safe_float(x["mic"]))
    summary["efficacy"] = effs[:2]

    # Extract Toxicity (First valid entry)
    for tox in data.get("hemoliticCytotoxicActivities") or []:
        if not isinstance(tox, dict): continue
        
        cell_dict = tox.get("targetCell") or {}
        cell = cell_dict.get("name")
        
        unit_dict = tox.get("unit") or {}
        
        if cell and tox.get("concentration"):
            summary["toxicity"].append({
                "cell_type": cell,
                "concentration": tox.get("concentration"),
                "unit": unit_dict.get("name", ""),
                "effect": tox.get("activityMeasureForLysisValue", "Unknown")
            })
            break

    return summary

def run_blast_and_parse(fasta_path, db_path, output_xml):
    """Executes BLASTP and returns parsed hit info."""
    try:
        subprocess.run([
            "blastp", "-query", fasta_path, "-db", db_path,
            "-outfmt", "5", "-max_target_seqs", "1", # Limiting to top hit for speed
            "-task", "blastp", "-num_threads", "8", "-out", output_xml, "-evalue", "10.0", 
        ], check=True, capture_output=True)
        
        with open(output_xml, "r") as f:
            records = list(NCBIXML.parse(f))
            if not records or not records[0].alignments:
                return None
            
            top_aln = records[0].alignments[0]
            top_hsp = top_aln.hsps[0]
            return {
                "accession": top_aln.accession,
                "identity": round((top_hsp.identities / top_hsp.align_length) * 100, 2),
                "e_value": top_hsp.expect
            }
    except Exception as e:
        logging.error(f"BLAST failed for {fasta_path}: {e}")
        return None

def Verify_DBAASP(input_data: dict) -> str:
    folder_path = input_data.get("folder_path")
    if not folder_path or not os.path.exists(folder_path):
        raise ValueError("Valid folder_path is required")

    csv_path = os.path.join(folder_path, "generated_sequences.csv")
    db_path = "/home/raymondlab/Documents/AMP-Agent/src/tools/Verifying/dbaasp/dbaasp"
    id_map_path = "/home/raymondlab/Documents/AMP-Agent/src/tools/Verifying/dbaasp/dbaasp_id.csv"
    hits_dir = os.path.join(folder_path, 'dbaasp_hits')
    
    os.makedirs(hits_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Load ID map once to avoid repeated disk reads
    id_df = pd.read_csv(id_map_path)
    id_df['id'] = id_df['id'].astype(str)
    id_map = dict(zip(id_df['id'], id_df['sequence']))

    filter_cols = [c for c in df.columns if "filter" in c]
    full_report_log = []
    dbaasp_results_col = []

    for _, row in df.iterrows():
        # Skip if filters not met

        if not all(row.get(col) == 1 for col in filter_cols):
            dbaasp_results_col.append("")
            continue
        # dbaasp_report exists, and row['dbaasp_report' is not empty, skip
        if 'dbaasp_report' in df.columns and (row.get('dbaasp_report')) and row['dbaasp_report']!="new" :
            dbaasp_results_col.append(row['dbaasp_report'])
            continue

        fasta = row["fasta_path"]
        xml_out = os.path.join(hits_dir, os.path.basename(fasta).replace(".fasta", ".xml"))
        
        hit = run_blast_and_parse(fasta, db_path, xml_out)
        
        if hit:
            acc = str(hit['accession'])
            seq = id_map.get(acc)
            
            report_data = "No DBAASP details found"
            if seq:
                info = fetch_dbaasp(f"https://dbaasp.org/peptides?sequence.value={seq}")
                
                # Check that info is actually a dict before using .get()
                if isinstance(info, dict) and info.get('data'):
                    first_item = info['data'][0]
                    
                    if isinstance(first_item, dict):
                        peptide_id = first_item.get('dbaaspId')
                        details = fetch_dbaasp(f"https://dbaasp.org/peptides/{peptide_id}")
                        
                        if details:
                            summary = parse_llm_ready_summary(details)
                            report_data = json.dumps(summary)
            
            log_entry = f"Seq: {row['sequence']} | Match: {acc} | ID: {hit['identity']}%"
            full_report_log.append(log_entry)
            dbaasp_results_col.append(report_data)
        else:
            dbaasp_results_col.append("No BLAST matches")
    if len(full_report_log) == 0:
        return "No sequences passed all filters."
    df['dbaasp_report'] = dbaasp_results_col

    reported = [col for col in df.columns if "report" in col]
    reported_done = []
    for col in reported:
        if col in df.columns:
            check = df[col].apply(lambda x: True if (x is not None) and (x != "")  and (x!="new") else False)
        if check.sum() == len(check):
            reported_done.append([col, len(check)])
    report = "Completed Verification :"

    for col in reported_done:
        report += f" {col[0]} ({col[1]} sequences completed),"
    
    full_report_log.append(report)
    df.to_csv(csv_path, index=False)


    return "\n".join(full_report_log)
