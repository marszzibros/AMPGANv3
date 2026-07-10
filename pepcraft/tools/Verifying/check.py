# from Bio.Blast import NCBIWWW, NCBIXML
# from Bio import Entrez, SeqIO
# import time
# import subprocess
# def search_peptide_homology(peptide_sequence):
#     """
#     Runs BLASTP against the NCBI 'nr' database and evaluates the top hit.
#     Flags as 'not novel' if the sequence identity is 100%.
#     """
#     print(f"Running BLAST for sequence (length {len(peptide_sequence)})... this may take a few minutes.")
#     subprocess.run([
#             "blastp", 
#             "-query", "peptides.fasta", 
#             "-db", "swissprot/swissprot",
#             "-outfmt", "5",              # Crucial: 5 outputs XML so Biopython can parse it
#             "-max_target_seqs", "3",
#             "-task", "blastp",           # Crucial: Standard blastp covers broader/longer sequences
#             "-num_threads", "8",
#             "-out", "hits.xml"
#         ], check=True) # check=True ensures Python throws an error if BLAST fails
#     exit()
#     # 2. Parse the XML Output
#     parsed_results = []
    
#     with open("hits.xml", "r") as result_handle:
#         # We use .parse() instead of .read() because a .fasta file might have multiple sequences
#         blast_records = NCBIXML.parse(result_handle)
        
#         for blast_record in blast_records:
#             query_id = blast_record.query
            
#             # Check if we got any alignments
#             if not blast_record.alignments:
#                 print(f"[{query_id}] No matches found.")
#                 parsed_results.append({"query": query_id, "error": "No matches"})
#                 continue
            
#             # Get the top hit (most homologous)
#             top_alignment = blast_record.alignments[0]
#             top_hsp = top_alignment.hsps[0] # High-scoring Segment Pair
            
#             # Calculate identity percentage
#             identity_pct = (top_hsp.identities / top_hsp.align_length) * 100
            
#             # Determine novelty
#             is_novel = True
#             if identity_pct == 100.0:
#                 is_novel = False
                
#             hit_info = {
#                 "query": query_id,
#                 "title": top_alignment.title,
#                 "accession": top_alignment.accession,
#                 "length": top_alignment.length,
#                 "e_value": top_hsp.expect,
#                 "identity_pct": round(identity_pct, 2),
#                 "is_novel": is_novel,
#                 # Biopython splits the alignment into 3 strings: query, match (the middle symbols), and subject
#                 "alignment_query": top_hsp.query,
#                 "alignment_match": top_hsp.match, 
#                 "alignment_subject": top_hsp.sbjct
#             }
            
#             parsed_results.append(hit_info)
            
#             # Print a quick summary to the console
#             print(f"[{query_id}] Top Hit: {hit_info['accession']} | Identity: {hit_info['identity_pct']}% | Novel: {hit_info['is_novel']}")
            
#     return parsed_results

# def get_taxonomy_and_functional_text(accession_id, email):
#     """
#     Fetches the GenBank record for a given accession ID to extract taxonomy,
#     description text, and potential GO terms.
#     """
#     # NCBI requires an email address to use their Entrez servers
#     Entrez.email = email
    
#     print(f"\nFetching record for Accession: {accession_id}...")
    
#     try:
#         # Fetch the protein record from the Entrez Protein database in GenBank format
#         handle = Entrez.efetch(db="protein", id=accession_id, rettype="gb", retmode="text")
#         record = SeqIO.read(handle, "genbank")
#         handle.close()
        
#         # Extract Taxonomy
#         taxonomy = record.annotations.get("taxonomy", [])
        
#         # Extract general description text
#         description = record.description
        
#         # Search through features for GO terms and product names
#         go_terms = []
#         products = []
#         for feature in record.features:
#             if "db_xref" in feature.qualifiers:
#                 for xref in feature.qualifiers["db_xref"]:
#                     if xref.startswith("GO:"):
#                         go_terms.append(xref)
            
#             if "product" in feature.qualifiers:
#                 products.extend(feature.qualifiers["product"])
                
#         # Remove duplicates from lists
#         go_terms = list(set(go_terms))
#         products = list(set(products))
        
#         info = {
#             "taxonomy": taxonomy,
#             "description": description,
#             "products_identified": products,
#             "go_terms": go_terms
#         }
        
#         return info

#     except Exception as e:
#         print(f"Error fetching data from NCBI: {e}")
#         return None

# # ==========================================
# # Example Usage
# # ==========================================
# if __name__ == "__main__":
#     # Example short peptide sequence 
#     test_sequence = "MKTIIALSYIFCLVFADYKDDDDK" 
    
#     # IMPORTANT: Replace with your actual email! NCBI will block you otherwise.
#     my_email = "jay-hwasung.jung@uvm.com" 
    
#     # 1. Run the BLAST search
#     blast_results = search_peptide_homology(test_sequence)
#     print(blast_results)
    
#     if blast_results:
        
#         # 2. Fetch Taxonomy and Database text
#         # Give NCBI servers a brief rest between requests (good practice)
#         time.sleep(1) 
        
#         db_info = get_taxonomy_and_functional_text(blast_results[0]['accession'], my_email)
        
#         if db_info:
#             print("\n--- Taxonomy & Functional Info ---")
#             print(f"Taxonomy: {' -> '.join(db_info['taxonomy'])}")
#             print(f"Description: {db_info['description']}")
#             print(f"Products: {', '.join(db_info['products_identified'])}")
#             print(f"GO Terms Found: {', '.join(db_info['go_terms']) if db_info['go_terms'] else 'None directly linked in GenBank feature tags'}")
import pandas as pd

df = pd.read_csv("/home/raymondlab/Documents/AMP-Agent/output_single_one_3_20/generated_sequences.csv")

reported = [col for col in df.columns if "report" in col]
reported_done = []
for col in reported:
    if col in df.columns:
        check = df[col].apply(lambda x: True if (x is not None) and (x != "")  and (x!="new") else False)
    if check.sum() == len(check):
        reported_done.append(col)


print(reported_done)