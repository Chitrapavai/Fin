import os
import json
import pdfplumber
import pandas as pd
import camelot
import numpy as np
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from google import genai
from google.genai.types import EmbedContentConfig
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.utils import embedding_functions

# API Setup
GEMINI_API_KEY =  "AIzaSyDqAYD6aPGt9FNGI55rBFQDqPdcjwTNnCg" #"AIzaSyDg4-xN4ItnZ0d6gyu3rQWpgt8Pt0MogNw"
client = genai.Client(api_key=GEMINI_API_KEY)

top_k = 3
top_k1 = 5
pdf_path = r"C:\Projects\Financial-Population\source-pdf\tfl-2024.pdf"
query_excel_path = r"C:\Projects\Financial-Population\question\tfl_abc_2024.xlsx"
source_sheet_name = "Sheet1"
output_csv_path = "tfl_direct_chroma_output.csv"

base_name = Path(pdf_path).stem
os.makedirs("embedding_tfl-2024_vdb3", exist_ok=True)
FULL_CONTENT_JSON_PATH = f"embedding_tfl-2024_vdb3/{base_name}_full_content.json"
EXCEL_FILE_STREAM = f"embedding_tfl-2024_vdb3/{base_name}_stream.xlsx"
EXCEL_FILE_LATTICE = f"embedding_tfl-2024_vdb3/{base_name}_lattice.xlsx"
JSON_FILE_STREAM = f"embedding_tfl-2024_vdb3/{base_name}_stream.json"
JSON_FILE_LATTICE = f"embedding_tfl-2024_vdb3/{base_name}_lattice.json"
JSON_FULL_CONTENT_METADATA = f"embedding_tfl-2024_vdb3/{base_name}_metadata.json"
NOTES_EXCEL_PATH = f"embedding_tfl-2024_vdb3/{base_name}_AllNotes.xlsx"
NOTE_EMBEDDINGS_CACHE = f"embedding_tfl-2024_vdb3/{base_name}_note_embeddings.json"
CHROMA_DIR = f"embedding_tfl-2024_vdb3/chroma_{base_name}"  # ChromaDB setup
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name=f"{base_name}_collection")

HEADER_LOOKUP = {
    "Balance Sheet": [
        "Notes to the Financial Statements",
        "Group Balance Sheet"
    ],
    "Consolidated Statement of Financial Position": [
        "Notes to the Financial Statements",
        "Group Balance Sheet"
    ],
    "Consolidated Balance Sheet": [
        "Notes to the Financial Statements",
        "Group Balance Sheet"
    ],
    "Group and Parent Company Balance Sheet": [
        "Notes to the Financial Statements",
        "Group Balance Sheet",
        "Corporation Balance Sheet"
    ],
    "Consolidated and Company Balance sheet": [
        "Notes to the Financial Statements",
        "Group Balance Sheet"
    ],
    "Company Balance Sheet": [
        "Notes to the Financial Statements",
        "Corporation Balance Sheet"
    ],
    "Consolidated Balance Sheet": [
        "Notes to the Financial Statements",
        "Group Balance Sheet"
    ]
}

def clean_chunks(chunks):
    return [chunk.strip() for chunk in chunks if isinstance(chunk, str) and chunk.strip()]

def process_excel_to_json(excel_file_path):
    try:
        excel_data = pd.ExcelFile(excel_file_path)
        json_data = {}
        for sheet_name in excel_data.sheet_names:
            df = excel_data.parse(sheet_name)
            json_data[sheet_name] = df.to_dict(orient="records")
        return json_data
    except Exception as e:
        print(f"Error processing Excel file '{excel_file_path}': {e}")
        return {}

def extract_notes_mapping_from_pdf(pdf_path):
    notes, current_note, current_text = {}, None, ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text: continue
            for line in text.splitlines():
                match = re.match(r"Note\s+(\d+)[\s:\-]", line.strip(), re.IGNORECASE)
                if match:
                    if current_note:
                        notes[current_note] = {"page": page.page_number, "content": current_text.strip()}
                    current_note = f"Note {match.group(1)}"
                    current_text = line
                elif current_note:
                    current_text += " " + line
        if current_note:
            notes[current_note] = {"page": len(pdf.pages), "content": current_text.strip()}
    return notes

def generate_google_embeddings(contents):
    if not contents or not isinstance(contents, list) or all(not content.strip() for content in contents):
        raise ValueError("Contents must be a non-empty list of non-empty strings.")
    try:
        embeddings = []
        for i in range(0, len(contents), 100):
            batch = contents[i:i + 100]
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=batch,
                config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            batch_embeddings = [emb.values for emb in response.embeddings]
            embeddings.extend(batch_embeddings)
            '''
            # Store each batch in Chroma - newly added
            for idx, (text, emb) in enumerate(zip(batch, batch_embeddings)):
                collection.add(
                    ids=[f"{len(collection.get()['ids']) + idx}"],
                    embeddings=[emb],
                    documents=[text],
                    metadatas={"source": base_name, "batch_index": i}
                )'''

        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

def extract_page_header_from_chunk(chunk: str) -> str:
    match = re.search(r"Page Header:\s*(.+)", chunk, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    return None

def normalize(text):
    return re.sub(r'[^a-z0-9 ]+', '', text.lower().strip())

def get_lookup_keys_for_header(header, threshold=70):
    normalized_input = normalize(header)
    best_score = 0
    best_key = None
    for key in HEADER_LOOKUP:
        normalized_key = normalize(key)
        score = fuzz.token_set_ratio(normalized_input, normalized_key)
        #print(f"Matching '{normalized_input}' with lookup key '{normalized_key}' → Score: {score}")
        if score >= threshold and score > best_score:
            best_score = score
            best_key = key
    if best_key:
        return HEADER_LOOKUP[best_key]
    return []

def search_google_embeddings(query, top_k, filter_header=None, threshold=70):
    
    try:
        #Build metadata filter for Chroma query
        metadata_filter = {}
        matched_headers = []

        if filter_header:
            print(f" Filtering by header: {filter_header}")

            # get all unique page_headers from Chroma
            try:
                all_meta = collection.get(include=["metadatas"])["metadatas"]
                unique_headers = {m.get("Page Header", "").strip() for m in all_meta if m.get("Page Header")}
                unique_headers = [h for h in unique_headers if h]
            except Exception as e:
                #print(f" Could not load page_header metadata from Chroma: {e}")
                unique_headers = []

            #  Direct match
            for h in unique_headers:
                score = fuzz.token_set_ratio(h.lower(), filter_header.lower())
                if score >= threshold:
                    matched_headers.append(h)
            '''
            if matched_headers:
                print(f" Direct page_header matches: {matched_headers}")
            '''

            #  HEADER_LOOKUP fallback 
            #if not matched_headers:
            #print(f"Trying HEADER_LOOKUP for: {filter_header}...")
            lookup_keys = get_lookup_keys_for_header(filter_header, threshold)
            if lookup_keys:
                for target in lookup_keys:
                    for h in unique_headers:
                        score = fuzz.token_set_ratio(h.lower(), target.lower())
                        if score >= threshold:
                            if h not in matched_headers:
                                matched_headers.append(h)
                print(f" HEADER_LOOKUP matches: {matched_headers}")
            else:
                print(f" No match in HEADER_LOOKUP for: {filter_header}")

            if not matched_headers:
                #print(f" No matching headers found for: {filter_header}")
                return []

            # Build OR filter for Chroma
            metadata_filter["$or"] = [{"Page Header": {"$eq": h}} for h in matched_headers]

        # Generate query embedding
        query_embedding = generate_google_embeddings([query])[0]

        # Query Chroma with metadata filter
        chroma_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=metadata_filter if metadata_filter else {}
        )

        # Format results
        results = []
        for doc, score, meta in zip(
            chroma_results.get("documents", [[]])[0],
            chroma_results.get("distances", [[]])[0],
            chroma_results.get("metadatas", [[]])[0]
        ):
            results.append({
                "chunk": doc,
                "score": float(score),
                "metadata": meta
            })

        return results

    except Exception as e:
        print(f" Error during ChromaDB retrieval for query '{query}': {e}")
        return []

def rank_and_format_chunks(chunks_b, chunks_c, top_n=4):
    combined = chunks_b + chunks_c
    combined_sorted = sorted(combined, key=lambda x: x["score"], reverse=True)
    top_chunks = combined_sorted[:top_n]
    return "\n\n---\n\n".join([f"Score: {round(chunk['score'], 4)}\n{chunk['chunk']}" for chunk in top_chunks])

def extract_pdf_chunks(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "**".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=40)
    return clean_chunks(splitter.split_text(full_text))

def main():
    print("=== PDF Embed & Retrieve Questions from Excel (B & C Columns) ===")

     # notes_map = extract_notes_mapping_from_pdf(pdf_path)

    if not os.path.exists(JSON_FILE_STREAM) or not os.path.exists(JSON_FILE_LATTICE):
        print("Processing PDF and extracting tables...")
        
        with pdfplumber.open(pdf_path) as pdf_plumber:
            pages_to_extract = pdf_plumber.pages[96:255]
            pdf_pages = [page.extract_text() for page in pages_to_extract]
             # notes_map = extract_notes_mapping_from_pdf(pdf_path)
    

        chroma_document = []
        chroma_metadata = []
        chunk_id  =[]
        # Flavor - Stream
        with pd.ExcelWriter(EXCEL_FILE_STREAM, engine='xlsxwriter') as writer:
            stream_tables = camelot.read_pdf(pdf_path, flavor="stream", pages='97-255')
            for i, table in enumerate(stream_tables):
                page_num = table.page
                table_index = i + 1
                table_df = table.df

                # table header extraction
                table_header = ', '.join(table_df.iloc[0].dropna().tolist()) if not table_df.empty else "N/A"
                
                # page level header from pdfplumber
                page_text = pdf_pages[page_num - 97] if page_num >= len(pdf_pages) else ""
                page_lines = page_text.splitlines() if page_text else []
                page_header = " | ".join(page_lines[:3]) if page_lines else "N/A"
                notes_found = re.findall(r"Note\s+\d+", table_df.to_string())
                # notes_texts = [notes_map.get(n, "") for n in notes_found]

                metadata = pd.DataFrame({
                    "Metadata": [
                        f"Page: {page_num}",
                        f"Table Index: {table_index}",
                        f"Table Header: {table_header}",
                        f"Page Header: {page_header}",
                        f"Notes Mentioned: {', '.join(notes_found) or 'None'}"
                    ]
                })
                new_table = pd.concat([metadata, table_df], ignore_index=True)
                new_table.to_excel(writer, sheet_name=f"Stream_Table_{i+1}", index=False, header=False)

                '''
                metadata_for_chroma = {
                    "Metadata": [
                        f"Page: {page_num}",
                        f"Table Index: {table_index}",
                        f"Table Header: {table_header}",
                        f"Page Header: {page_header}",
                        f"Notes Mentioned: {', '.join(notes_found) or 'None'}"
                    ]
                }
                '''
                metadata_for_chroma = {
                    
                        "Page": page_num,
                        "Table Index": table_index,
                        "Table Header": table_header,
                        "Page Header": page_header,
                        "Notes Mentioned": f"{', '.join(notes_found) or 'None'}"
                    
                }
                #table_text = table.df.to_String()
                #chroma_document.append(table_text)
                chroma_metadata.append(metadata_for_chroma)
                chunk_id.append(f"Stream_Table_{i+1}")
        # Flavor - Lattice
        with pd.ExcelWriter(EXCEL_FILE_LATTICE, engine='xlsxwriter') as writer:
            lattice_tables = camelot.read_pdf(pdf_path, flavor="lattice", pages='97-255')
            for i, table in enumerate(lattice_tables):
                page_num = table.page
                table_index = i + 1
                table_df = table.df

                # table header extraction
                table_header = ', '.join(table_df.iloc[0].dropna().tolist()) if not table_df.empty else "N/A"
                
                # page header extraction from pdfplumber
                page_text = pdf_pages[page_num - 97] if page_num >= len(pdf_pages) else ""
                page_lines = page_text.splitlines() if page_text else []
                page_header = " | ".join(page_lines[:3]) if page_lines else "N/A"
                notes_found = re.findall(r"Note\s+\d+", table_df.to_string())
                metadata = pd.DataFrame({
                    "Metadata": [
                        f"Page: {page_num}",
                        f"Table Index: {table_index}",
                        f"Table Header: {table_header}",
                        f"Page Header: {page_header}",
                        f"Notes Mentioned: {', '.join(notes_found) or 'None'}"
                    ]
                })
                new_table = pd.concat([metadata, table_df], ignore_index=True)
                new_table.to_excel(writer, sheet_name=f"Lattice_Table_{i+1}", index=False, header=False)

                # metadata_for_chroma = {
                #     "Metadata": [
                #         f"Page: {page_num}",
                #         f"Table Index: {table_index}",
                #         f"Table Header: {table_header}",
                #         f"Page Header: {page_header}",
                #         f"Notes Mentioned: {', '.join(notes_found) or 'None'}"
                #     ]
                # }
                metadata_for_chroma = {
                    
                        "Page": page_num,
                        "Table Index": table_index,
                        "Table Header": table_header,
                        "Page Header": page_header,
                        "Notes Mentioned": f"{', '.join(notes_found) or 'None'}"
                    
                }
                # table_text = table.df.to_String()
                # chroma_document.append(table_text)
                chroma_metadata.append(metadata_for_chroma)
                chunk_id.append(f"Lattice_Table_{i+1}")
        json.dump(process_excel_to_json(EXCEL_FILE_STREAM), open(JSON_FILE_STREAM, 'w', encoding='utf-8'), indent=4)
        json.dump(process_excel_to_json(EXCEL_FILE_LATTICE), open(JSON_FILE_LATTICE, 'w', encoding='utf-8'), indent=4)
        chroma_data = {
            "metadata" : chroma_metadata,
            "id" : chunk_id
        }
        
    
        json.dump(chroma_data, open(JSON_FULL_CONTENT_METADATA, 'w', encoding='utf-8'), indent = 4)
        
        '''
            # Example: show first few chunks with metadata
            for d in docs[:3]:
                print(f"Page: {d['metadata']['page_number']} | Header: {d['metadata']['page_header']}")
                print(d["content"][:200], "...\n")
                '''


    if os.path.exists(FULL_CONTENT_JSON_PATH):
        with open(FULL_CONTENT_JSON_PATH, 'r', encoding='utf-8') as f:
            full_content_json = json.load(f)
            table_chunks1 = full_content_json.get("stream_tables", [])
            table_chunks2 = full_content_json.get("lattice_tables", [])
            docs = full_content_json.get("docs", [])
            chroma_metadata = full_content_json.get("metadata",{})
            chunks = table_chunks1 + table_chunks2 + docs
            embeddings = generate_google_embeddings(chunks)  # Chroma stores inside this

            collection.add(
                            embeddings=embeddings,
                            documents=chunks,
                            metadatas= chroma_metadata["metadata"],
                            ids = chroma_metadata["id"]

                        )
            
    else:
        table_chunks1 = clean_chunks([json.dumps(table) for table in json.load(open(JSON_FILE_STREAM)).values()])
        table_chunks2 = clean_chunks([json.dumps(table) for table in json.load(open(JSON_FILE_LATTICE)).values()])
        with open(JSON_FULL_CONTENT_METADATA, "r", encoding="utf-8") as f:
            chroma_data = json.load(f)
        '''
        with pdfplumber.open(pdf_path) as pdf:
            page_data_list = []
        
            for i, page in enumerate(pdf.pages, start = 134):
                page_text = page.extract_text()
                if not page_text:
                    continue
                
                # Page header = first line of page text
                page_header = page_text.split("\n")[0].strip() if "\n" in page_text else page_text.strip()
                
                page_data_list.append({
                    "text": page_text,
                    "page_number": i,
                    "page_header": page_header
                })

            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=40)
            text_chunk = []
            #page_metadata = []
            for page_data in page_data_list:
                chunks = splitter.split_text(page_data["text"])
                i =1
                for chunk in chunks:
                    text_chunk.append(chunk)
                    chroma_metadata.append(
                       {
                            "page_number": page_data["page_number"],
                            "page_header": page_data["page_header"]
                        }
                    )
                    page_no = page_data["page_number"]
                    chunk_id.append(f"Page_text_{page_no}_{i}")
                    i = i +1
        '''
        page_data_list = []
        with open(FULL_CONTENT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump({"stream_tables": table_chunks1, "lattice_tables": table_chunks2, "docs": page_data_list, "metadata": chroma_data}, f, indent=4)

        
    
    # Processing queru from excel
    df = pd.read_excel(query_excel_path, sheet_name=source_sheet_name)
    #df = df[df['METRIC'].isin(['Marketable Investments', 'Tax Recoverable'])]
    # df = df[0:100]
    results = []
    for index, row in df.iterrows():
        sheet_name = row['METRIC SHEET NAME']
        metric = row['METRIC']
         #question_b = str(row["QUESTION1"]).strip() if "QUESTION1" in row and pd.notna(row["QUESTION1"]) else ""
        question_c = str(row["QUESTION2"]).strip() if "QUESTION2" in row and pd.notna(row["QUESTION2"]) else ""
        
        # chunks_b = search_google_embeddings(question_b, chunks, embeddings, top_k1) if question_b else []
        # chunks_c = search_google_embeddings(question_c, chunks, embeddings, top_k1) if question_c else []
        
        
        filter_header = str(row["STATEMENT TYPE"]).strip() if "STATEMENT TYPE" in row and pd.notna(row["STATEMENT TYPE"]) else None
        #chunks_b = search_google_embeddings(question_b, chunks, embeddings, top_k, filter_header=filter_header)
        chunks_c = search_google_embeddings(question_c, top_k, filter_header=filter_header)

        
        #formatted_b = "\n---\n".join([c["chunk"] for c in chunks_b])
        formatted_c = "\n---\n".join([c["chunk"] for c in chunks_c])

        

        results.append({
            "Sheet name": sheet_name,
            "Metric": metric,
            #"Question B": question_b,
            #"Retrieved B": formatted_b,
            # "Relevant Notes B": top_notes_b,
            "Question C": question_c,
            "Retrieved C": formatted_c
            # "Relevant Notes C": top_notes_c,
            # "Top Ranked Chunks (B+C)": ranked_combined,
            # "Top Related Notes": top_notes
        })

    
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=False)
    print(f"\n Retrieved chunks and notes saved in the following path: {output_csv_path}")

if __name__ == "__main__":
    main()
