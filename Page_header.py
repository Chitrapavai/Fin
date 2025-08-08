import pandas as pd

def extract_page_headers_from_excel(file_path):
    headers = set()
    xls = pd.ExcelFile(file_path, engine='openpyxl')
    
    for sheet_name in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, nrows=5, header=None)
            for row in df.itertuples(index=False):
                for cell in row:
                    if isinstance(cell, str) and "page header" in cell.lower():
                        headers.add(cell.strip())
        except Exception as e:
            print(f"Error reading sheet '{sheet_name}' in {file_path}: {e}")
    return sorted(headers)

def main():
    # Replace with the actual paths to your Excel files
    stream_path = r"C:\Projects\Financial-Population\embedding_Bristol_2024_new\Bristol_2024_stream.xlsx"
    lattice_path = r"C:\Projects\Financial-Population\embedding_Bristol_2024_new\Bristol_2024_lattice.xlsx"

    stream_headers = extract_page_headers_from_excel(stream_path)
    lattice_headers = extract_page_headers_from_excel(lattice_path)

    # Convert to DataFrames
    df_stream = pd.DataFrame(stream_headers, columns=["Page Header"])
    df_lattice = pd.DataFrame(lattice_headers, columns=["Page Header"])

    # Save to a single Excel file with two sheets
    output_file = "bristol_page_headers_combined.xlsx"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_stream.to_excel(writer, sheet_name='Stream', index=False)
        df_lattice.to_excel(writer, sheet_name='Lattice', index=False)

    print(f"Page headers saved to '{output_file}' with separate sheets for Stream and Lattice.")

if __name__ == "__main__":
    main()
