import pandas as pd
import numpy as np
import evaluate

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def get_references(df):
    '''
    This function takes a dataframe and return references from the given dataframe 
    and returns it in two different formats for evaluation.
    '''
    references = df['reference'].tolist()
    references_bleu = [[reference] for reference in references]

    return [references, references_bleu]


def get_predictions(df):
    '''
    This function takes a dataframe and return predictions from the given dataframe.
    '''
    predictions = df['prediction'].tolist()
    return predictions


test_dataset_extended = pd.read_csv('data_preprocessing/qa_evaluation/approach1/references_predictions.csv', delimiter='\t', dtype=str)

all_full_types = ['Confirmation', 'Factoid-type', 'List-type', 'Causal', 'Hypothetical', 'Complex']

# Having one df that has all the original question types
df_all = test_dataset_extended[test_dataset_extended['question_type'].isin(all_full_types)]

# Breaking the test-set into smaller sets based on the question type
df_confirmation = test_dataset_extended[test_dataset_extended['question_type'] == 'Confirmation']

df_factoid_type = test_dataset_extended[test_dataset_extended['question_type'] == 'Factoid-type']

df_list_type = test_dataset_extended[test_dataset_extended['question_type'] == 'List-type']

df_causal = test_dataset_extended[test_dataset_extended['question_type'] == 'Causal']

df_hypothetical = test_dataset_extended[test_dataset_extended['question_type'] == 'Hypothetical']

# Here We're getting the sets for different types of Complex generations in terms of their generation
df_complex = test_dataset_extended[test_dataset_extended['question_type'] == 'Complex'] # this is for all types of complex questions

df_complex_dense = test_dataset_extended[test_dataset_extended['question_type'] == 'complex_dense']

df_complex_sparse = test_dataset_extended[test_dataset_extended['question_type'] == 'complex_sparse']

df_complex_2chunks = test_dataset_extended[test_dataset_extended['question_type'] == 'complex_2chunks']

df_complex_3chunks = test_dataset_extended[test_dataset_extended['question_type'] == 'complex_3chunks']


# Printing the sizes of subsets
print(f"Sizes:\nNumber of All Questions: {len(df_all)}")
print(f"Number of Confirmation Questions: {len(df_confirmation)}\nNumber of Factoid-type Questions: {len(df_factoid_type)}")
print(f"Number of List-type Questions: {len(df_list_type )}\nNumber of Causal Questions: {len(df_causal)}")
print(f"Number of Hypothetical Questions: {len(df_hypothetical)}\nNumber of Complex Questions: {len(df_complex)}")
print(f"Number of Complex Questons (Generated using Dense Search for Similariy Search): {len(df_complex_dense)}")
print(f"Number of Complex Questons (Generated using Sparse Search for Similariy Search): {len(df_complex_sparse)}")
print(f"Number of Complex Questons (Generated using Two Similar Chunks): {len(df_complex_2chunks)}")
print(f"Number of Complex Questons (Generated using Three Similar Chunks): {len(df_complex_3chunks)}\n")


# Here we get the references and prediction for all questions for overall evaluation
references_all = get_references(df_all)[0]
references_bleu_all = get_references(df_all)[1]
predictions_all = get_predictions(df_all)


# Here we get the references and pedictions for each type of question for evaluation
references_confirmation = get_references(df_confirmation)[0]
references_bleu_confirmation = get_references(df_confirmation)[1]
predictions_confirmation = get_predictions(df_confirmation)

references_factoid_type = get_references(df_factoid_type)[0]
references_bleu_factoid_type = get_references(df_factoid_type)[1]
predictions_factoid_type = get_predictions(df_factoid_type)

references_list_type = get_references(df_list_type)[0]
references_bleu_list_type = get_references(df_list_type)[1]
predictions_list_type= get_predictions(df_list_type)

references_causal = get_references(df_causal)[0]
references_bleu_causal = get_references(df_causal)[1]
predictions_causal = get_predictions(df_causal)

references_hypothetical = get_references(df_hypothetical)[0]
references_bleu_hypothetical = get_references(df_hypothetical)[1]
predictions_hypothetical = get_predictions(df_hypothetical)

references_complex = get_references(df_complex)[0]
references_bleu_complex = get_references(df_complex)[1]
predictions_complex = get_predictions(df_complex)

references_complex_dense = get_references(df_complex_dense)[0]
references_bleu_complex_dense = get_references(df_complex_dense)[1]
predictions_complex_dense = get_predictions(df_complex_dense)

references_complex_sparse = get_references(df_complex_sparse)[0]
references_bleu_complex_sparse = get_references(df_complex_sparse)[1]
predictions_complex_sparse = get_predictions(df_complex_sparse)

references_complex_2chunks = get_references(df_complex_2chunks)[0]
references_bleu_complex_2chunks = get_references(df_complex_2chunks)[1]
predictions_complex_2chunks = get_predictions(df_complex_2chunks)

references_complex_3chunks = get_references(df_complex_3chunks)[0]
references_bleu_complex_3chunks = get_references(df_complex_3chunks)[1]
predictions_complex_3chunks = get_predictions(df_complex_3chunks)

# print(test_dataset_extended.tail())


# # HERE WE ARE COMPUTING BLEU RESULTS 
bleu_results_all = bleu.compute(predictions=predictions_all, references=references_bleu_all)

bleu_results_confirmation = bleu.compute(predictions=predictions_confirmation, references=references_bleu_confirmation)
bleu_results_factoid_type = bleu.compute(predictions=predictions_factoid_type, references=references_bleu_factoid_type)
bleu_results_list_type = bleu.compute(predictions=predictions_list_type, references=references_bleu_list_type)
bleu_results_causal = bleu.compute(predictions=predictions_causal, references=references_bleu_causal)
bleu_results_hypothetical = bleu.compute(predictions=predictions_hypothetical, references=references_bleu_hypothetical)

bleu_results_complex = bleu.compute(predictions=predictions_complex, references=references_bleu_complex)
bleu_results_complex_dense = bleu.compute(predictions=predictions_complex_dense, 
                                          references=references_bleu_complex_dense)
bleu_results_complex_sparse = bleu.compute(predictions=predictions_complex_sparse, 
                                           references=references_bleu_complex_sparse)
bleu_results_complex_2chunks = bleu.compute(predictions=predictions_complex_2chunks, 
                                            references=references_bleu_complex_2chunks)
bleu_results_complex_3chunks = bleu.compute(predictions=predictions_complex_3chunks,
                                            references=references_bleu_complex_3chunks)

# PRINTING THE BLEU SCORES
print("BLEU Scores:\n")
print(f"BLEU Score for All Questions: {bleu_results_all['bleu']}, Precision-1: {bleu_results_all['precisions'][0]}, \
Precision-2: {bleu_results_all['precisions'][1]}, Precision-3: {bleu_results_all['precisions'][2]}, \
Precision-4: {bleu_results_all['precisions'][3]}\n")

print(f"BLEU Score for Confirmation Questions: {bleu_results_confirmation['bleu']}, \
Precision-1: {bleu_results_confirmation['precisions'][0]}, Precision-2: {bleu_results_confirmation['precisions'][1]}, \
Precision-3: {bleu_results_confirmation['precisions'][2]}, Precision-4: {bleu_results_confirmation['precisions'][3]}\n")

print(f"BLEU Score for Factoid Type Questions: {bleu_results_factoid_type['bleu']}, \
Precision-1: {bleu_results_factoid_type['precisions'][0]}, Precision-2: {bleu_results_factoid_type['precisions'][1]}, \
Precision-3: {bleu_results_factoid_type['precisions'][2]}, Precision-4: {bleu_results_factoid_type['precisions'][3]}\n")

print(f"BLEU Score for List Type Questions: {bleu_results_list_type['bleu']}, \
Precision-1: {bleu_results_list_type['precisions'][0]}, Precision-2: {bleu_results_list_type['precisions'][1]}, \
Precision-3: {bleu_results_list_type['precisions'][2]}, Precision-4: {bleu_results_list_type['precisions'][3]}\n")

print(f"BLEU Score for Causal Questions: {bleu_results_causal['bleu']}, \
Precision-1: {bleu_results_causal['precisions'][0]}, Precision-2: {bleu_results_causal['precisions'][1]}, \
Precision-3: {bleu_results_causal['precisions'][2]}, Precision-4: {bleu_results_causal['precisions'][3]}\n")

print(f"BLEU Score for Hypothetical Questions: {bleu_results_hypothetical['bleu']}, \
Precision-1: {bleu_results_hypothetical['precisions'][0]}, Precision-2: {bleu_results_hypothetical['precisions'][1]}, \
Precision-3: {bleu_results_hypothetical['precisions'][2]}, Precision-4: {bleu_results_hypothetical['precisions'][3]}\n")

print(f"BLEU Score for All Complex Questions: {bleu_results_complex['bleu']}, \
Precision-1: {bleu_results_complex['precisions'][0]}, Precision-2: {bleu_results_complex['precisions'][1]}, \
Precision-3: {bleu_results_complex['precisions'][2]}, Precision-4: {bleu_results_complex['precisions'][3]}\n")

print(f"BLEU Score for Complex Questions Generated using Dense Seach: {bleu_results_complex_dense['bleu']}, \
Precision-1: {bleu_results_complex_dense['precisions'][0]}, Precision-2: {bleu_results_complex_dense['precisions'][1]}, \
Precision-3: {bleu_results_complex_dense['precisions'][2]}, Precision-4: {bleu_results_complex_dense['precisions'][3]}\n")

print(f"BLEU Score for Complex Questions Generated using Sparse Search: {bleu_results_complex_sparse['bleu']}, \
Precision-1: {bleu_results_complex_sparse['precisions'][0]}, Precision-2: {bleu_results_complex_sparse['precisions'][1]}, \
Precision-3: {bleu_results_complex_sparse['precisions'][2]}, Precision-4: {bleu_results_complex_sparse['precisions'][3]}\n")

print(f"BLEU Score for Complex Questions Generated using Two Chunks: {bleu_results_complex_2chunks['bleu']}, \
Precision-1: {bleu_results_complex_2chunks['precisions'][0]}, Precision-2: {bleu_results_complex_2chunks['precisions'][1]}, \
Precision-3: {bleu_results_complex_2chunks['precisions'][2]}, Precision-4: {bleu_results_complex_2chunks['precisions'][3]}\n")

print(f"BLEU Score for Complex Questions Generated using Three Chunks: {bleu_results_complex_3chunks['bleu']}, \
Precision-1: {bleu_results_complex_3chunks['precisions'][0]}, Precision-2: {bleu_results_complex_3chunks['precisions'][1]}, \
Precision-3: {bleu_results_complex_3chunks['precisions'][2]}, Precision-4: {bleu_results_complex_3chunks['precisions'][3]}\n")


# HERE WE ARE COMPUTING ROUGE RESULTS 
rouge_results_all = rouge.compute(predictions=predictions_all, 
                                           references=references_all)

rouge_results_confirmation = rouge.compute(predictions=predictions_confirmation, 
                                           references=references_confirmation)
rouge_results_factoid_type = rouge.compute(predictions=predictions_factoid_type, 
                                           references=references_factoid_type)
rouge_results_list_type = rouge.compute(predictions=predictions_list_type, 
                                           references=references_list_type)
rouge_results_causal = rouge.compute(predictions=predictions_causal, 
                                           references=references_causal)
rouge_results_hypothetical = rouge.compute(predictions=predictions_hypothetical, 
                                           references=references_hypothetical)

rouge_results_complex = rouge.compute(predictions=predictions_complex, 
                                           references=references_complex)
rouge_results_complex_dense = rouge.compute(predictions=predictions_complex_dense, 
                                           references=references_complex_dense)
rouge_results_complex_sparse = rouge.compute(predictions=predictions_complex_sparse, 
                                           references=references_complex_sparse)
rouge_results_complex_2chunks = rouge.compute(predictions=predictions_complex_2chunks, 
                                           references=references_complex_2chunks)
rouge_results_complex_3chunks = rouge.compute(predictions=predictions_complex_3chunks, 
                                           references=references_complex_3chunks)

# print(rouge_results_complex_3chunks)
# PRINTING THE ROUGE SCORES
print(f"ROUGE Scores for All Questions - ROUGE-1: {rouge_results_all['rouge1']}, \
ROUGE-2: {rouge_results_all['rouge2']}, ROUGE-L: {rouge_results_all['rougeL']}, \
ROUGE-Lsum: {rouge_results_all['rougeLsum']}")

print(f"ROUGE Scores for Confirmation Questions - ROUGE-1: {rouge_results_confirmation['rouge1']}, \
ROUGE-2: {rouge_results_confirmation['rouge2']}, ROUGE-L: {rouge_results_confirmation['rougeL']}, \
ROUGE-Lsum: {rouge_results_confirmation['rougeLsum']}")

print(f"ROUGE Scores Factoid Type Questions - ROUGE-1: {rouge_results_factoid_type['rouge1']}, \
ROUGE-2: {rouge_results_factoid_type['rouge2']}, ROUGE-L: {rouge_results_factoid_type['rougeL']}, \
ROUGE-Lsum: {rouge_results_factoid_type['rougeLsum']}")
# print("BLEU Score for Factoid Type Questions:", bleu_results_factoid_type['bleu'])
# print("BLEU Score for List Type Questions:", bleu_results_list_type['bleu'])
# print("BLEU Score for Causal Questions:", bleu_results_causal['bleu'])
# print("BLEU Score for Hypothetical Questions:", bleu_results_hypothetical['bleu'])
# print("BLEU Score for All Complex Questions:", bleu_results_complex['bleu'])
# print("BLEU Score for Complex Questions Generated using Dense Seach:", bleu_results_complex_dense['bleu'])
# print("BLEU Score for Complex Questions Generated using Sparse Search:", bleu_results_complex_sparse['bleu'])
# print("BLEU Score for Complex Questions Generated using Two Chunks:", bleu_results_complex_2chunks['bleu'])
# print("BLEU Score for Complex Questions Generated using Three Chunks:", bleu_results_complex_3chunks['bleu'])



# bertscore_results_all = bertscore.compute(predictions=predictions_all, 
#                                           references=references_all, lang="en")
# precision_scores = bertscore_results_all['precision']
# recall_scores = bertscore_results_all['recall']
# f1_scores = bertscore_results_all['f1']

# print(np.mean(f1_scores))
