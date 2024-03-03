import pandas as pd
import requests
import evaluate


def get_references(df):
    '''
    This function takes a dataframe and return references from the given dataframe 
    in two different formats for evaluation.
    '''
    references = df['answer'].tolist()
    references_bleu = [[reference] for reference in references]

    return [references, references_bleu]


def get_prediction_from_llm(question):
    url = f'http://localhost:8000/retrieve_documents_dense?query_str={question}'
    
    response = requests.get(url)

    if not response.ok:
        raise ValueError(f'HTTP error! Status: {response.status_code}')

    return response.json()['message']


def modify_original_prediciton(original_predicion):
    '''
    This function is used to clean the original prediction by removing sources from it.
    As our references do not contain sources.
    '''
    index = original_predicion.find("_|") # find the index of "_|"

    # If "Sources:" is found, remove everything starting from that index
    if index != -1: # if "Sources:" is present in the original prediction
        modified_prediciton = original_predicion[:index] # we keep everyhing until "_|"
    
    else:
        modified_prediciton = original_predicion # else keep the original prediction

    return modified_prediciton


def get_predictions(df):
    predictions = []
    for question in df['question']:
        # print(question)
        original_prediction = get_prediction_from_llm(question)
        prediction = modify_original_prediciton(original_prediction)
        predictions.append(prediction)
    return predictions

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load('meteor')
bertscore = evaluate.load("bertscore")

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

# SOME DEBUGGING CODE HERE
# print(get_predictions(df_complex))
# print(get_predictions(df_complex))
# print(get_predictions(df_confirmation))
prediction = get_prediction_from_llm("How does chiral analysis of synthetic cathinones in forensic laboratories contribute to drug intelligence and the development of better treatment for overdose or addiction?")
print(modify_original_prediciton(prediction))
# print(get_references(df_confirmation)[0])
# print(get_references(df_confirmation)[1])


# HERE WE ARE COMPUTING BLEU RESULTS 
bleu_results_confirmation = bleu.compute(predictions=get_predictions(df_confirmation), references=get_references(df_confirmation)[1])
bleu_results_factoid_type = bleu.compute(predictions=get_predictions(df_factoid_type), references=get_references(df_factoid_type)[1])
bleu_results_list_type = bleu.compute(predictions=get_predictions(df_list_type), references=get_references(df_list_type)[1])
bleu_results_causal = bleu.compute(predictions=get_predictions(df_causal), references=get_references(df_causal)[1])
bleu_results_hypothetical = bleu.compute(predictions=get_predictions(df_hypothetical), references=get_references(df_hypothetical)[1])

bleu_results_complex = bleu.compute(predictions=get_predictions(df_complex), references=get_references(df_complex)[1])
bleu_results_complex_dense = bleu.compute(predictions=get_predictions(df_complex_dense), references=get_references(df_complex_dense)[1])
bleu_results_complex_sparse = bleu.compute(predictions=get_predictions(df_complex_sparse), references=get_references(df_complex_sparse)[1])
bleu_results_complex_2chunks = bleu.compute(predictions=get_predictions(df_complex_2chunks), references=get_references(df_complex_2chunks)[1])
bleu_results_complex_3chunks = bleu.compute(predictions=get_predictions(df_complex_3chunks), references=get_references(df_complex_3chunks)[1])


# HERE WE ARE COMPUTING ROUGE RESULTS
rouge_results_confirmation = rouge.compute(predictions=get_predictions(df_confirmation), 
                                           references=get_references(df_confirmation)[0])
rouge_results_factoid_type = rouge.compute(predictions=get_predictions(df_factoid_type), 
                                           references=get_references(df_factoid_type)[0])
rouge_results_list_type = rouge.compute(predictions=get_predictions(df_list_type), 
                                           references=get_references(df_list_type)[0])
rouge_results_causal = rouge.compute(predictions=get_predictions(df_causal), 
                                           references=get_references(df_causal)[0])
rouge_results_hypothetical = rouge.compute(predictions=get_predictions(df_hypothetical), 
                                           references=get_references(df_hypothetical)[0])

rouge_results_complex = rouge.compute(predictions=get_predictions(df_complex), 
                                           references=get_references(df_complex)[0])
rouge_results_complex_dense = rouge.compute(predictions=get_predictions(df_complex_dense), 
                                           references=get_references(df_complex_dense)[0])
rouge_results_complex_sparse = rouge.compute(predictions=get_predictions(df_complex_sparse), 
                                           references=get_references(df_complex_sparse)[0])
rouge_results_complex_2chunks = rouge.compute(predictions=get_predictions(df_complex_2chunks), 
                                           references=get_references(df_complex_2chunks)[0])
rouge_results_complex_3chunks = rouge.compute(predictions=get_predictions(df_complex_3chunks), 
                                           references=get_references(df_complex_3chunks)[0])

# HERE WE ARE COMPUTING BERTScore
bertscore_results_confirmation = bertscore.compute(predictions=get_predictions(df_confirmation), 
                                                   references=get_references(df_confirmation)[0], lang="en")
bertscore_results_factoid_type = bertscore.compute(predictions=get_predictions(df_factoid_type), 
                                                   references=get_references(df_factoid_type)[0], lang="en")
bertscore_results_list_type = bertscore.compute(predictions=get_predictions(df_list_type), 
                                                   references=get_references(df_list_type)[0], lang="en")
bertscore_results_causal = bertscore.compute(predictions=get_predictions(df_causal), 
                                                   references=get_references(df_causal)[0], lang="en")
bertscore_results_hypothetical = bertscore.compute(predictions=get_predictions(df_hypothetical), 
                                                   references=get_references(df_hypothetical)[0], lang="en")

bertscore_results_complex = bertscore.compute(predictions=get_predictions(df_complex), 
                                                   references=get_references(df_complex)[0], lang="en")
bertscore_results_complex_dense = bertscore.compute(predictions=get_predictions(df_complex_dense), 
                                                   references=get_references(df_complex_dense)[0], lang="en")
bertscore_results_complex_sparse = bertscore.compute(predictions=get_predictions(df_complex_sparse), 
                                                   references=get_references(df_complex_sparse)[0], lang="en")
bertscore_results_complex_2chunks = bertscore.compute(predictions=get_predictions(df_complex_2chunks), 
                                                   references=get_references(df_complex_2chunks)[0], lang="en")
bertscore_results_complex_3chunks = bertscore.compute(predictions=get_predictions(df_complex_3chunks), 
                                                   references=get_references(df_complex_3chunks)[0], lang="en")
