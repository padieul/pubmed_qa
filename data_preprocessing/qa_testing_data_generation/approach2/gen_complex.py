import pandas as pd
import ast
import openai

# Assuming you've set your OpenAI API key elsewhere in the environment for security reasons
openai.api_key = 'YOUR_API_KEY'

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset='pmid')
    df['key_words'] = df['key_words'].fillna('[]')
    df['key_words_list'] = df['key_words'].apply(lambda x: ast.literal_eval(x))

    pmid_pairs = []
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i < j:
                common_keywords = set(row_i['key_words_list']) & set(row_j['key_words_list'])
                if len(common_keywords) > 10:
                    pmid_pairs.append([row_i['pmid'], row_j['pmid']])
    return pmid_pairs

def generate_complex_questions(abstract1, abstract2):
    prompt = (f"Create a complex question that requires multi-part reasoning by understanding the semantics of multiple text snippets from two abstracts."
              f"Abstract 1: {abstract1}\n\n"
              f"Abstract 2: {abstract2}\n\n"
              f"Generate a question and provide an answer based on these abstracts:")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Adjusted for the correct model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1500,
        n=1,  # Number of completions to generate
        stop=["\n\n"]
    )

    return response.choices[0].message['content'].strip()

def get_abstract(pmid):
    # Placeholder function to retrieve abstract text by PMID
    # Implement this function based on your data retrieval method
    return "Abstract text goes here..."

def main(csv_path, output_csv_path):
    pmid_pairs = preprocess_data(csv_path)
    results = []

    for pair in pmid_pairs:
        abstract1 = get_abstract(pair[0])
        abstract2 = get_abstract(pair[1])
        complex_question_and_answer = generate_complex_questions(abstract1, abstract2)
        results.append([pair[0], pair[1], abstract1, abstract2, complex_question_and_answer])

    results_df = pd.DataFrame(results, columns=['pmid1', 'pmid2', 'abstract1', 'abstract2', 'Complex Question and Answer'])
    print(results_df)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Output saved to {output_csv_path}")

# Adjust the paths as necessary
csv_path = 'data_preprocessing\\data\\processed_data_for_testing_500_100.csv'
output_csv_path = 'data_preprocessing\\qa_testing_data_generation\\approach1\\Complex_Questions.csv'

# Run the main function
main(csv_path, output_csv_path)
