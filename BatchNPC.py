import pandas as pd
import grequests
import urllib.parse
import json
from tqdm import tqdm

SERVER_URL = "http://localhost:6541"

def read_smiles_file(file_path: str, smiles_col: str = "smiles", sep: str = "\t") -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=sep)
    # Try to find the column case-insensitively
    actual_col = next((col for col in df.columns if col.lower() == smiles_col.lower()), None)
    if actual_col is None:
        raise KeyError(f"No column named '{smiles_col}' found in {file_path}")
    df.rename(columns={actual_col: "smiles"}, inplace=True)
    return df

def generate_urls(df: pd.DataFrame) -> list:
    urls = []
    for entry in tqdm(df.to_dict(orient="records")):
        smiles = str(entry["smiles"])
        if len(smiles) > 5:
            encoded = urllib.parse.quote(smiles)
            urls.append(f"{SERVER_URL}/classify?smiles={encoded}")
    return urls

def send_async_requests(urls: list, batch_size: int = 20) -> list:
    rs = (grequests.get(u) for u in urls)
    responses = grequests.map(rs, size=batch_size)
    return responses

def parse_responses(responses: list) -> list:
    parsed = []
    for resp in responses:
        if resp and resp.status_code == 200:
            try:
                parsed.append(resp.json())
            except json.JSONDecodeError:
                parsed.append({"error": "invalid json"})
        else:
            parsed.append({"error": "no response"})
    return parsed

def save_results(df_input: pd.DataFrame, parsed_results: list, output_file: str):
    df_results = pd.DataFrame(parsed_results)
    df_final = pd.concat([df_input, df_results], axis=1)
    df_final.to_csv(output_file, index=False)
    print(f"Batch classification complete. Output saved to '{output_file}'")

def main():
    input_file = "SMILES.csv"
    output_file = "npclassifier_output.csv"
    
    try:
        df = read_smiles_file(input_file, sep="\t")
        urls = generate_urls(df)
        responses = send_async_requests(urls)
        results = parse_responses(responses)
        save_results(df, results, output_file)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
