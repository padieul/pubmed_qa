import pandas as pd
import csv
import requests


def create_references_predictions_csv(file_path):
    '''
    This function is used to create a csv file where we store our 
    questions, references, predictions and types of the question.
    It is only executed once.
    '''
    with open(file_path, 'w', newline='') as file:
        # Create a CSV writer object with tab as the delimiter
        csv_writer = csv.writer(file, delimiter='\t')  # Specify tab as the delimiter

        header = ["question", "reference", "prediction", "question_type"]
        csv_writer.writerow(header)

    return

def write_to_csv(file_path, question, reference, prediction, question_type):
    '''
    This function is used to write a record to our references_prediction.csv file.
    '''
    with open(file_path, 'a', newline='') as file:
        csv_writer = csv.writer(file, delimiter='\t')
                        
        new_record = [question, reference, prediction, question_type]
                
        csv_writer.writerow(new_record)
    return


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


def get_prediction_from_llm(question):
    '''
    This function is used to get the prediction from our llm for the given question.
    We also modify it here - remove the sources as our references do not contain sources.
    '''
    url = f'http://localhost:8000/retrieve_documents_dense?query_str={question}'
    
    response = requests.get(url)

    if not response.ok:
        raise ValueError(f'HTTP error! Status: {response.status_code}')

    original_prediction = response.json()['message']
    return modify_original_prediciton(original_prediction)


# run once to create the file
# create_references_predictions_csv("data_preprocessing/qa_evaluation/approach1/references_predictions.csv")

# here we are getting the test dataset
test_set = pd.read_csv('data_preprocessing/qa_testing_data_generation/approach1/test_dataset.csv', delimiter='\t', dtype=str)
# print(test_set.head())

file_path_refs_preds = "data_preprocessing/qa_evaluation/approach1/references_predictions.csv"

for index in test_set.index:
    record = test_set.loc[index]
    question = record['question']
    reference = record['answer']
    prediction = get_prediction_from_llm(question)
    question_type = record['question_type']
    
    write_to_csv(file_path_refs_preds, question, reference, prediction, question_type)

    if question_type == "Complex":
        # Below we also have 4 different complex question types that essentially have the same content 
        # However, these question types are essential to know how our model performs in different scenarios
        if record['similarity_search'] == "Sparse": # Sparse search is used while question generation
            write_to_csv(file_path_refs_preds, question, reference, prediction, "complex_sparse")
        elif record['similarity_search'] == "Dense":
            write_to_csv(file_path_refs_preds, question, reference, prediction, "complex_dense")

        if record['pmid2'].notna() and record['pmid3'].isna():
            write_to_csv(file_path_refs_preds, question, reference, prediction, "complex_2chunks")
        elif record['pmid2'].notna() and record['pmid3'].notna():
            write_to_csv(file_path_refs_preds, question, reference, prediction, "complex_3chunks")