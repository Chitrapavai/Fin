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

# API Setup
GEMINI_API_KEY = "AIzaSyDg4-xN4ItnZ0d6gyu3rQWpgt8Pt0MogNw"   # mithra - API Key
client = genai.Client(api_key=GEMINI_API_KEY)

top_k = 3
top_k1 = 5

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
            if not hasattr(response, 'embeddings') or not response.embeddings:
                raise RuntimeError("Embedding API response is empty. Possible cause: invalid API key or quota exceeded.")
            embeddings.extend([emb.values for emb in response.embeddings])
        return embeddings
    except Exception as e:
        raise RuntimeError(f"❌ Error generating embeddings: {e}")


def search_google_embeddings(query, chunks, chunk_embeddings, top_k):
    chunks = clean_chunks(chunks)
    if not chunks:
        print(f"No valid chunks to search for query: {query}")
        return []
    try:
        query_embedding = generate_google_embeddings([query])[0]
        similarities = np.dot(chunk_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [{"chunk": chunks[i], "score": float(similarities[i])} for i in top_indices]
    except Exception as e:
        print(f"Error during embedding search for query '{query}': {e}")
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
    pdf_path = r"C:/Projects/Financial-Population/source-pdf/Bristol_2024.pdf"
    query_excel_path = r"C:/Projects/Financial-Population/question/bristol_questions_2024.xlsx"
    source_sheet_name = "direct"
    output_csv_path = "bristol_without-header_direct-op.csv"
    base_name = Path(pdf_path).stem
    os.makedirs("embedding_Bristol_2024_new", exist_ok=True)
    EMBEDDINGS_CACHE = f"embedding_Bristol_2024_new/{base_name}_embedded_text.json"
    EXCEL_FILE_STREAM = f"embedding_Bristol_2024_new/{base_name}_stream.xlsx"
    EXCEL_FILE_LATTICE = f"embedding_Bristol_2024_new/{base_name}_lattice.xlsx"
    JSON_FILE_STREAM = f"embedding_Bristol_2024_new/{base_name}_stream.json"
    JSON_FILE_LATTICE = f"embedding_Bristol_2024_new/{base_name}_lattice.json"
    NOTES_EXCEL_PATH = f"embedding_Bristol_2024_new/{base_name}_AllNotes.xlsx"
    NOTE_EMBEDDINGS_CACHE = f"embedding_Bristol_2024_new/{base_name}_note_embeddings.json"
    '''
    notes_map = extract_notes_mapping_from_pdf(pdf_path)

    if not os.path.exists(JSON_FILE_STREAM) or not os.path.exists(JSON_FILE_LATTICE):
        print("Extracting tables from PDF...")

        with pd.ExcelWriter(EXCEL_FILE_STREAM) as writer:
            stream_tables = camelot.read_pdf(pdf_path, flavor="stream", pages='all')
            for i, table in enumerate(stream_tables):
                page_num = table.page
                table_df = table.df
                metadata = pd.DataFrame({"Metadata": [f"Page: {page_num}"]})
                new_table = pd.concat([metadata, table_df], ignore_index=True)
                new_table.to_excel(writer, sheet_name=f"Stream_Table_{i+1}", index=False, header=False)

        with pd.ExcelWriter(EXCEL_FILE_LATTICE) as writer:
            lattice_tables = camelot.read_pdf(pdf_path, flavor="lattice", pages='all')
            for i, table in enumerate(lattice_tables):
                page_num = table.page
                table_df = table.df
                metadata = pd.DataFrame({"Metadata": [f"Page: {page_num}"]})
                new_table = pd.concat([metadata, table_df], ignore_index=True)
                new_table.to_excel(writer, sheet_name=f"Lattice_Table_{i+1}", index=False, header=False)

        json.dump(process_excel_to_json(EXCEL_FILE_STREAM), open(JSON_FILE_STREAM, 'w', encoding='utf-8'), indent=4)
        json.dump(process_excel_to_json(EXCEL_FILE_LATTICE), open(JSON_FILE_LATTICE, 'w', encoding='utf-8'), indent=4)
    '''
    if os.path.exists(EMBEDDINGS_CACHE):
        with open(EMBEDDINGS_CACHE, 'r', encoding='utf-8') as f:
            cached_embeddings = json.load(f)
            table_chunks1 = cached_embeddings.get("stream_tables", [])
            table_chunks2 = cached_embeddings.get("lattice_tables", [])
            docs = cached_embeddings.get("docs", [])
    else:
        table_chunks1 = clean_chunks([json.dumps(table) for table in json.load(open(JSON_FILE_STREAM)).values()])
        table_chunks2 = clean_chunks([json.dumps(table) for table in json.load(open(JSON_FILE_LATTICE)).values()])
        with pdfplumber.open(pdf_path) as pdf:
            document_texts = "**".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        docs = clean_chunks(RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=40).split_text(document_texts))
        with open(EMBEDDINGS_CACHE, 'w', encoding='utf-8') as f:
            json.dump({"stream_tables": table_chunks1, "lattice_tables": table_chunks2, "docs": docs}, f, indent=4)

    # Combine all chunks
    chunks = docs + table_chunks1 + table_chunks2
    embeddings = generate_google_embeddings(chunks)
    '''
    if os.path.exists(NOTE_EMBEDDINGS_CACHE):
        with open(NOTE_EMBEDDINGS_CACHE, 'r', encoding='utf-8') as f:
            note_embeddings = json.load(f)
    else:
        print("Building note embeddings...")
        print("Total notes in notes_map:", len(notes_map))
    
    valid_notes = [
        (k, notes_map[k]) for k in notes_map
        if notes_map[k].get('content') and notes_map[k]['content'].strip()
    ]

    print("Valid notes count:", len(valid_notes))

    contents = [note['content'] for _, note in valid_notes]
    if not contents:
        raise ValueError("❌ No valid note contents found to generate embeddings.")

    try:
        emb_list = generate_google_embeddings(contents)
    except Exception as e:
        raise RuntimeError(f"❌ Embedding failed: {e}")

    note_embeddings = {
        k: {'embedding': emb, 'page': note['page']}
        for (k, note), emb in zip(valid_notes, emb_list)
    }

    with open(NOTE_EMBEDDINGS_CACHE, 'w', encoding='utf-8') as f:
        json.dump(note_embeddings, f, indent=4)
    '''
    # Process Questions from Excel
    df = pd.read_excel(query_excel_path, sheet_name=source_sheet_name)
    results = []
    for index, row in df.iterrows():
        sheet_name = row['METRIC SHEET NAME']
        metric = row['METRIC']
        id_value = row[0]
        # question_b = str(row["QUESTION1"]).strip() if "QUESTION1" in row and pd.notna(row["QUESTION1"]) else ""
        question_c = str(row["QUESTION2"]).strip() if "QUESTION2" in row and pd.notna(row["QUESTION2"]) else ""

        # chunks_b = search_google_embeddings(question_b, chunks, embeddings, top_k1) if question_b else []
        chunks_c = search_google_embeddings(question_c, chunks, embeddings, top_k1) if question_c else []

        # formatted_b = "\n---\n".join([c["chunk"] for c in chunks_b])
        formatted_c = "\n---\n".join([c["chunk"] for c in chunks_c])

        #ranked_combined = rank_and_format_chunks(chunks_b, chunks_c, top_n=4)

        results.append({
            "Sheet name": sheet_name,
            "Metric": metric,
            # "Question B": question_b,
            # "Retrieved B": formatted_b,
            "Question C": question_c,
            "Retrieved C": formatted_c,
            #"Top Ranked Chunks (B+C)": ranked_combined
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Retrieved chunks saved to: {output_csv_path}")

if __name__ == "__main__":
    main()
