import pandas as pd
import spacy
from collections import Counter

def tokenize_description(text, nlp):
    """
    Process the full text with spaCy to obtain sentence boundaries and full context.
    Then, for each sentence, iterate over its tokens (with full context)
    and filter out punctuation, whitespace, and stop words.
    Each token is then lemmatized using better_lemmatize().
    
    This sentence-by-sentence processing ensures that each token’s POS is determined
    in the context of its entire sentence, improving disambiguation.
    Returns a list of lowercased lemmas.
    """
    # Process the full text
    doc = nlp(text)
    tokens = []
    # Iterate over sentences to ensure each token is evaluated in its full-sentence context
    for sent in doc.sents:
        for token in sent:
            if token.is_punct or token.is_space or token.is_stop:
                continue
            tokens.append(token.lemma_.lower())
    return tokens

def tokenize_adjective_cell(cell, nlp):
    """
    Splits a cell (assumed to contain comma-separated adjectives), then for each candidate:
      1. Process the candidate text with spaCy to obtain full sentence context.
      2. Iterate over tokens and select only adjectives (POS == "ADJ") that are not stop words or punctuation.
      3. Lemmatize the selected tokens using better_lemmatize().
    Returns a list of lowercased lemmas.
    """
    tokens = []
    candidates = [item.strip() for item in cell.split(",") if item.strip()]
    for cand in candidates:
        doc = nlp(cand)
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "ADJ" and not token.is_stop and not token.is_punct:
                    tokens.append(token.lemma_.lower())
    return tokens

def tokenize_named_entities(text, nlp):
    """
    Process the text with spaCy to extract named entities.
    Returns a list of entity texts as they appear.
    """
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def compute_counters(data, columns, tokenize_func, nlp):
    """
    For each specified column in the DataFrame, tokenize the cell text (using tokenize_func)
    and update a frequency Counter with the resulting tokens.
    Returns a dictionary mapping each column name to its Counter.
    """
    counters = {col: Counter() for col in columns}
    for _, row in data.iterrows():
        for col in columns:
            cell = row[col]
            if pd.isna(cell):
                continue
            tokens = tokenize_func(cell, nlp)
            counters[col].update(tokens)
    return counters

def get_sorted_list(counter):
    """
    Return a list of (token, frequency) tuples sorted in descending order by frequency.
    """
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)

def build_master_dataframe(counters, columns):
    """
    Build a master DataFrame that aggregates sorted frequency data for each column.
    For each column, two output columns are created: {col}_Word and {col}_Frequency.
    Additionally, overall frequency columns ("Overall_Word" and "Overall_Frequency")
    are computed across all specified columns.
    Lists are padded so that every output column has the same length.
    """
    overall = Counter()
    sorted_data = {}
    max_length = 0

    for col in columns:
        sorted_list = get_sorted_list(counters[col])
        sorted_data[col] = sorted_list
        overall.update(counters[col])
        max_length = max(max_length, len(sorted_list))
    overall_sorted = get_sorted_list(overall)
    max_length = max(max_length, len(overall_sorted))

    df_dict = {}
    for col in columns:
        words = [item[0] for item in sorted_data[col]]
        freqs = [item[1] for item in sorted_data[col]]
        while len(words) < max_length:
            words.append("")
            freqs.append(0)
        df_dict[f"{col}_Word"] = words
        df_dict[f"{col}_Frequency"] = freqs

    overall_words = [item[0] for item in overall_sorted]
    overall_freqs = [item[1] for item in overall_sorted]
    while len(overall_words) < max_length:
        overall_words.append("")
        overall_freqs.append(0)
    df_dict["Overall_Word"] = overall_words
    df_dict["Overall_Frequency"] = overall_freqs

    return pd.DataFrame(df_dict)

def process_and_save(data, columns, tokenize_func, nlp, output_filename):
    """
    Compute frequency counters for the given columns, build a master DataFrame,
    and save it as a CSV file.
    """
    counters = compute_counters(data, columns, tokenize_func, nlp)
    master_df = build_master_dataframe(counters, columns)
    master_df.to_csv(output_filename, index=False)
    print(f"Saved: {output_filename}")

if __name__ == "__main__":
    # Load data from the Excel file.
    data = pd.read_excel('survey_results.xlsx', engine='openpyxl')
    
    data.columns = [
        "Timestamp",
        "Description_1", "Adjectives_1",
        "Description_2", "Adjectives_2",
        "Description_3", "Adjectives_3",
        "Description_4", "Adjectives_4",
        "Description_5", "Adjectives_5",
        "Description_6", "Adjectives_6",
        "Familiarity_AI_Gen", "AI_Art_Real", "Education"
    ]
    description_cols = ["Description_1", "Description_2", "Description_3",
                        "Description_4", "Description_5", "Description_6"]
    adjective_cols = ["Adjectives_1", "Adjectives_2", "Adjectives_3",
                      "Adjectives_4", "Adjectives_5", "Adjectives_6"]

    # Load the spaCy model (which also sets up sentence segmentation).
    nlp = spacy.load('en_core_web_sm')

    # Process and save CSV files for descriptions and adjectives.
    process_and_save(data, description_cols, tokenize_description, nlp, "descriptions_frequencies.csv")
    process_and_save(data, adjective_cols, tokenize_adjective_cell, nlp, "adjectives_frequencies.csv")
