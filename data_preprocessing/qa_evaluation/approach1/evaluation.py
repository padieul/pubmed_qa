import pandas as pd
import evaluate


def get_references(df):
    '''
    This function takes a dataframe and return references from the given dataframe 
    in two different formats for evaluation.
    '''
    references = df['answer'].tolist()
    references_bleu = [[reference] for reference in references]

    return references, references_bleu

# here we are getting the test dataset for evaluation
test_set = pd.read_csv('data_preprocessing/qa_testing_data_generation/approach1/test_dataset.csv', delimiter='\t', dtype=str)
# print(test_set.head())

# Beaking the test-set into smaller sets based on the question type
df_confirmation = test_set[test_set['question_type'] == 'Confirmation']

df_factoid_type = test_set[test_set['question_type'] == 'Factoid-type']

df_list_type = test_set[test_set['question_type'] == 'List-type']

df_causal = test_set[test_set['question_type'] == 'Causal']

df_hypothetical = test_set[test_set['question_type'] == 'Hypothetical']


# Here We're getting the sets for different types of Complex generations in terms of their generation
df_complex = test_set[test_set['question_type'] == 'Complex']

df_complex_dense = df_complex[df_complex['similarity_search'] == 'Dense']

df_complex_sparse = df_complex[df_complex['similarity_search'] == 'Sparse']

df_complex_2chunks = df_complex[df_complex['pmid2'].notna() & df_complex['pmid3'].isna()]

df_complex_3chunks = df_complex[df_complex['pmid2'].notna() & df_complex['pmid3'].notna()]

# Printing the sizes of subsets
print(f"Sizes:\nNumber of Confirmation Questions: {len(df_confirmation)}\nNumber of Factoid-type Questions: {len(df_factoid_type)}")
print(f"Number of List-type Questions: {len(df_list_type )}\nNumber of Causal Questions: {len(df_causal)}")
print(f"Number of Hypothetical Questions: {len(df_hypothetical)}\nNumber of Complex Questions: {len(df_complex)}")
print(f"Number of Complex Questons (Generated using Dense Search for Similariy Search): {len(df_complex_dense)}")
print(f"Number of Complex Questons (Generated using Sparse Search for Similariy Search): {len(df_complex_sparse)}")
print(f"Number of Complex Questons (Generated using Two Similar Chunks): {len(df_complex_2chunks)}")
print(f"Number of Complex Questons (Generated using Three Similar Chunks): {len(df_complex_3chunks)}")
