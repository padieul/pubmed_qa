import pandas as pd
import ast
import openai

# Your OpenAI API key
openai.api_key = 'your_openai_api_key'

def safe_literal_eval(s):
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # Handle malformed strings; return an empty list or another default value
        return []

def generate_complex_question(abstract1, abstract2):
    prompt = (f"Create a complex question that requires multi-part reasoning by understanding "
              f"the semantics of two abstracts.\n\nAbstract 1: {abstract1}\n\nAbstract 2: {abstract2}\n\n")
    try:
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n"]
        )
        question = response.choices[0].text.strip()
        # Placeholder for answer since generating accurate answers would require another call or process
        answer = "Answer to be determined..."
        return question, answer
    except Exception as e:
        print(f"Error generating question: {e}")
        return "", ""

def extract_unique_pairs(input_csv_path, output_csv_path):
    # Load the CSV file, assuming no header; adjust if your file has a header
    df = pd.read_csv(input_csv_path, header=None, usecols=[0, 3, 5])
    
    # Assign column names for clarity
    df.columns = ['pmid', 'chunk', 'key_words']
    
    # Convert the 'key_words' column from string representation of lists to actual lists
    df['key_words_list'] = df['key_words'].apply(lambda x: safe_literal_eval(x))
    
    # Use a set to store already seen pairs and ensure uniqueness
    seen_pairs = set()
    results = []
    
    for i in range(len(df) - 1):
        for j in range(i + 1, len(df)):
            # Sort PMIDs to ensure uniqueness regardless of order
            pmid1 = df.at[i, 'pmid']
            pmid2 = df.at[j, 'pmid']
            if pmid1 != pmid2:  # Ensure the PMIDs are different to avoid identical pairs
                pmids = tuple(sorted([pmid1, pmid2]))
            
                if pmids not in seen_pairs:
                    common_keywords = set(df.at[i, 'key_words_list']) & set(df.at[j, 'key_words_list'])
                    if len(common_keywords) > 10:
                        seen_pairs.add(pmids)
                        results.append({
                            'pmid1': pmid1,
                            'pmid2': pmid2,
                            'chunk1': df.at[i, 'chunk'],
                            'chunk2': df.at[j, 'chunk']
                        })
    
    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)
    
    # Save the results to a CSV file
    results_df.to_csv(output_csv_path, index=False)

# Paths to your input and output CSV files
csv_path = 'data_preprocessing\\data\\processed_data_for_testing_500_100.csv'
output_csv_path = 'data_preprocessing\\qa_testing_data_generation\\approach2\\Complex_Questions.csv'

# Execute the function
extract_unique_pairs(csv_path, output_csv_path)
