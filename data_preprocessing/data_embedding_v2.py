import pandas as pd
from angle_emb import AnglE, Prompts
from tqdm import tqdm
import csv
import os, sys


# This script is to generate embeddings for the segmented abstracts using AnglE embeddings


def embed_data(source_file: str=None, destination_file: str=None, start_index: int=0, end_index: int=0) -> None:
    '''
    A helper function to embed the chunked data retrieved from PubMed

    Parameters:
        source_file (str): the location of the source csv file 
        destination_folder (str): the location of the produced file
        start_index (int): the start index in the input file to start embedding
        end_index (int): the end index in the input file where to stop embedding 
    '''

    # Source file is required
    if source_file is None:
        return
    
    # Import the data that we exported from PubMed
    df = pd.read_csv(source_file)

    # Drop abstracts with NaN values 
    df = df.dropna(subset=['abstract'])

    # Initialize AnglE embedding model
    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
    angle.set_prompt(prompt=Prompts.C)   

    rows_list = []

    for index, row in tqdm(df.iterrows(), total=end_index, desc='Embedding data'):    
    
        if index < start_index:
            continue
        if index == end_index:
            break        

        embedding = angle.encode({'text': row['chunk']})

        embedding = embedding[0].tolist()

        rows_list.append([row['pmid'], row['title'], row['abstract'], row['chunk_id'], row['chunk'], embedding, row['key_words'], row['authors'], row['journal'], row['year'], row['month'], row['source'], row['country']])


    df_save = pd.DataFrame(rows_list, columns=["pmid", "title", "abstract", "chunk_id", "chunk", "embedding", "key_words", "authors", "journal", "year", "month", "source", "country"])
    

    if os.path.exists(destination_file):
        df_save.to_csv(destination_file, mode='a', index=False, header=False)
    else:
        df_save.to_csv(destination_file,index=False)


if __name__ == "__main__":

    source_file = os.path.join(sys.path[0], "data\\data_chunks_500_250.csv") if os.name == "nt" else os.path.join(sys.path[0], "data/data_chunks_500_250.csv")
    destination_file = os.path.join(sys.path[0], "data\\data_embeddings_500_250.csv") if os.name == "nt" else os.path.join(sys.path[0], "data/data_embeddings_500_250.csv")

    
    embed_data(source_file=source_file, destination_file=destination_file, start_index=1001, end_index=2000)