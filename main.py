import pandas as pd
import numpy as np

from google import genai
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv

from tqdm import tqdm


# ---------- DataFrame utilities ----------

def remove_cols(df):
    columns_to_drop = ['created', 'id', 'new_price', 'pharmacology', 'route', 'updated']
    return df.drop(columns=columns_to_drop, errors='ignore')


def remove_duplicates(df):
    return df.drop_duplicates(subset=['tradename'])


def sort_by_name(df):
    return df.sort_values(by='tradename')


def chunking(arr, chunk_size):
    num_chunks = len(arr) // chunk_size
    chunks = arr[:num_chunks * chunk_size].reshape(num_chunks, chunk_size)
    return chunks, num_chunks


# ---------- Patch utilities ----------

def get_start_index(patches_dir="patches"):
    """Return the next patch index to write to, based on existing files."""
    if not os.path.exists(patches_dir):
        os.makedirs(patches_dir)
        return 0

    files = [f for f in os.listdir(patches_dir) if f.startswith("patch") and f.endswith(".csv")]

    if not files:
        return 0

    indices = []
    for f in files:
        try:
            num = int(f.replace("patch", "").replace(".csv", ""))
            indices.append(num)
        except:
            pass

    return max(indices) + 1 if indices else 0


def aggregate_patches():
    patches_dir = 'patches'
    output_file = 'output.csv'

    patch_files = [f for f in os.listdir(patches_dir) if f.endswith('.csv')]

    df_list = []
    for file in patch_files:
        df = pd.read_csv(os.path.join(patches_dir, file))
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(output_file, index=False)
    print(f"All patches have been aggregated into {output_file}")


# ---------- Pydantic Models ----------

class DrugItem(BaseModel):
    drug_name: str = Field(description="Clean medicine name only without dosage.")
    dose: str = Field(description="Extracted dosage value. If missing, return '0'.")


class DrugBatch(BaseModel):
    items: List[DrugItem]


# ---------- Gemini API ----------

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)


def extract_from_chunk(lines):
    joined_lines = "\n".join(lines)

    prompt = f"""
    You will receive a list of medicine lines.

    Your task is to extract EXACTLY three fields from each line:

    1. drug_name → The clean medicine name ONLY.
    2. dose → Must contain a number + a unit.

    Rules:
    - Always format doses correctly.
    - Missing items → return 'Null'.
    - Ignore 'f.c' and similar patterns.
    - If a number exists with no unit → treat as mg.
    - If the drug is only numbers or number+letters → treat it as drug_name.
    - Ignore variations of 'for'.

    Output must be clean JSON list.
    {joined_lines}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": DrugBatch.model_json_schema(),
        },
    )

    return DrugBatch.model_validate_json(response.text)


# ---------- Main script ----------

def add_original_name_to_existing_patches():
    """
    Run ONCE:
    Add original tradename column to old patches 
    that contain only (drug_name, dose, form).
    """
    patches_dir = "patches"
    df = pd.read_csv('source/Egypt-drugs-database/(CSV) New prices up to 03-08-2024.csv')

    # Clean the DF exactly like the main script
    df = remove_cols(df)
    df = remove_duplicates(df)
    df = sort_by_name(df)

    tradenames = np.array(df['tradename'])
    _, num_chunks = chunking(tradenames, 50)

    for i in range(num_chunks):
        patch_file = f"{patches_dir}/patch{i}.csv"
        if not os.path.exists(patch_file):
            continue

        patch_df = pd.read_csv(patch_file)

        # Add the original name for each row in the patch
        original_chunk = tradenames[i * 50 : (i + 1) * 50]

        # Resize to match rows count
        patch_df["original_name"] = original_chunk[:len(patch_df)]

        # Move original_name to first column
        cols = ["original_name"] + [c for c in patch_df.columns if c != "original_name"]
        patch_df = patch_df[cols]

        patch_df.to_csv(patch_file, index=False)

    print("All existing patches updated with original_name column.")



# ---------- Main script ----------

def main():

    # ----- Load & clean data -----
    df = pd.read_csv('source\missing_products_to_process.csv')
    # df = remove_cols(df)
    # df = remove_duplicates(df)
    df = sort_by_name(df)

    tradenames = np.array(df['tradename'])
    chunks, num_of_chunks = chunking(tradenames, chunk_size=50)

    # ----- Determine where to resume -----
    start_index = get_start_index()
    print(f"Resuming from patch index: {start_index}")

    # ----- Process chunks -----
    for i in tqdm(range(start_index, num_of_chunks)):
        print(f"Processing chunk {i} ...")

        lines = chunks[i].tolist()
        result = extract_from_chunk(lines)

        rows = [
            {
                "original_name": chunks[i][idx],          # NEW: original name
                "drug_name": item.drug_name,
                "dose": item.dose
            }
            for idx, item in enumerate(result.items)
        ]

        patch_df = pd.DataFrame(rows)
        patch_df.to_csv(f'patches/patch{i}.csv', index=False)

        print(f"Chunk {i} completed and saved.")


if __name__ == "__main__":
    main()
    # aggregate_patches()
