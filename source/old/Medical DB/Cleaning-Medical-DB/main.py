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


# ---------- Pydantic Models ----------

class DrugItem(BaseModel):
    drug_name: str = Field(description="Clean medicine name only without dosage.")
    dose: str = Field(description="Extracted dosage value. If missing, return '0'.")
    form: str = Field(description="Form of the drug. If missing, return '0'.")


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
    2. dose → Must contain a number + a unit (mg, ml, %, g, mcg, IU, mg/ml, etc.).
       - Always put a space between number and unit ("20 mg", "250 mg/5 ml").
       - If no dose exists, return 'Null'.
    3. form → The form of the drug (tablet, capsule, vial, syrup, gel, etc.).
       - If missing, return 'Null'.

    Rules:
    - Never mix units with the form.
    - Never return a number without a unit.
    - Never return a unit without a number.
    - If anything is missing → return 'Null'.
    - ignore words like "f.c", ..etc with its number preceding it.
    - if found number with no unit → make the unit mg by default, make sure to add a space between number and unit.
    - consider drugs that are consist of numbers only or number with characters e.g. 1 2 3 one two three as drug names.
    - ignroe 'for' with it's variations.

    Output must be clean JSON only, structured as a list of objects.

    Here are the lines:
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

def main():
    df = pd.read_csv('source/Egypt-drugs-database/(CSV) New prices up to 03-08-2024.csv')

    df = remove_cols(df)
    df = remove_duplicates(df)
    df = sort_by_name(df)

    tradenames = np.array(df['tradename'])
    chunks, num_of_chunks = chunking(tradenames, chunk_size=50)

    for i in tqdm(range(num_of_chunks)):
        print(f"Processing chunk {i} ...")

        lines = chunks[i].tolist()
        result = extract_from_chunk(lines)

        rows = [
            {"drug_name": item.drug_name, "dose": item.dose, "form": item.form}
            for item in result.items
        ]

        patch_df = pd.DataFrame(rows)
        patch_df.to_csv(f'patches/patch{i}.csv', index=False)

        print(f"Chunk {i} completed and saved.")


if __name__ == "__main__":
    main()
