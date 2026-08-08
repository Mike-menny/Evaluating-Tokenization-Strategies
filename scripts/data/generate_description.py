import argparse
import os
import time
import pandas as pd
from openai import OpenAI
from duckduckgo_search import DDGS

parser = argparse.ArgumentParser(description="Generate prompts for MAESTRO pieces using an LLM.")
parser.add_argument(
    "--max_sample",
    type=int,
    default=None,
    metavar="N",
    help="Process only the first N unique pieces (default: process all).",
)
args = parser.parse_args()

# 1. Configure API: OpenAI-compatible (Groq, DeepSeek, OpenAI, etc.)
# Set your API key and base_url; change model name to match the provider.
API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")  # Groq
MODEL_NAME = os.environ.get("OPENAI_MODEL", "deepseek-chat")  # Groq; e.g. deepseek-chat for DeepSeek

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 2. File path settings (MAESTRO dataset)
dataset_dir = '/root/maestro-v3.0.0'
input_file = os.path.join(dataset_dir, 'maestro-v3.0.0.csv')
output_file = os.path.join(dataset_dir, 'with_prompt.csv')

df_input = pd.read_csv(input_file)
output_columns = list(df_input.columns) + ['prompt']

# 3. Read or create progress
# Resume logic: one prompt per (composer, title); if script stops, next run continues
if os.path.exists(output_file):
    df_output = pd.read_csv(output_file)
    processed_pieces = set(zip(df_output['canonical_composer'], df_output['canonical_title']))
    print(f"Detected existing progress, {len(processed_pieces)} pieces done. Resuming...")
else:
    processed_pieces = set()
    df_input.iloc[:0].reindex(columns=output_columns).to_csv(
        output_file, index=False, encoding='utf-8-sig'
    )

# Unique (composer, title) pairs to process
unique_pieces = df_input.drop_duplicates(subset=['canonical_composer', 'canonical_title'])
if args.max_sample is not None:
    n = min(args.max_sample, len(unique_pieces))
    unique_pieces = unique_pieces.sample(n=n, random_state=None).reset_index(drop=True)
    print(f"Randomly selected {n} unique pieces (--max_sample={args.max_sample}).")
total_pieces = len(unique_pieces)

# Web search for factual grounding (RAG)
ddgs = DDGS()

# 4. Iterate and process (one API call per piece; same prompt for all rows of that piece)
for piece_idx, (_, piece_row) in enumerate(unique_pieces.iterrows(), start=1):
    composer = piece_row['canonical_composer']
    title = piece_row['canonical_title']
    piece_key = (composer, title)

    if piece_key in processed_pieces:
        continue

    print(f"Processing piece [{piece_idx}/{total_pieces}]: {composer} — {title}")

    # Step A: Web search for factual info (style, opus, era)
    search_query = f"{composer} {title} musical style analysis key signature opus"
    search_results_text = ""
    try:
        results = ddgs.text(search_query, max_results=3)
        for res in results:
            search_results_text += res.get("body", "") + "\n"
        print(f"   🔍 Fetched search context ({len(search_results_text)} chars)")
    except Exception as e:
        print(f"   ⚠️ Search failed: {e}. Using model knowledge only.")
        search_results_text = "No search results available."

    time.sleep(1.0)  # Brief pause after search to avoid rate limits

    # Step B: RAG prompt with search results
    system_prompt = """You are a strict musicologist. Write a highly accurate reverse-prompt based EXCLUSIVELY on the search results.
Rules:
1. Exact Genre: Match the genre from the title (e.g., Sonata, Poème). No hallucinations.
2. Opus Awareness: Respect the composer's specific period (Early/Middle/Late). Do not apply late-period traits to early works.
3. Strict Harmony: Never use "atonal" if a key signature exists or if the piece is merely highly chromatic.
4. No Stereotypes: Rely ONLY on the provided text."""

    user_prompt = f"""
Piece: '{title}' by '{composer}'
Search Results: {search_results_text}

Write a prompt under 100 words using this EXACT format:
"Give me a song that is [2-3 adjectives], composed in the [Early/Middle/Late] [Composer] style. It should be a [Exact Genre] in the [Era] era, featuring [rhythm/instrumentation]. Incorporate [texture/melody/harmony traits], and a transition from [mood 1] to [mood 2]."
"""

    # 5. Call API and handle errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            result = (response.choices[0].message.content or "").strip()

            # All rows for this (composer, title) get the same prompt; append to output CSV
            mask = (df_input['canonical_composer'] == composer) & (df_input['canonical_title'] == title)
            rows_to_append = df_input[mask].copy()
            rows_to_append['prompt'] = result
            rows_to_append.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')

            print(f"✅ Success: {result[:50]}...")
            break  # Success, exit retry loop

        except Exception as e:
            print(f"❌ Error: {e}. Retrying ({attempt + 1}/{max_retries})...")
            time.sleep(10)  # Wait longer after error

    # 6. Rate limit: brief pause after each request (tune for your provider)
    time.sleep(1.0)

print(f"🎉 Done! {total_pieces} piece(s) processed. Output: {output_file}")