import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import numpy as np
import os, sys

# This script is to segment the data we collected from PubMeb into chunks with custom size and overlap values and store the result in CSV file


def chunk_data(source_file: str=None, chunk_size: int=500, chunk_overlap: int=250, destination_folder: str=None) -> None:
    '''
    A helper function to chunk the data retrieved from PubMed

    Parameters:
        source_file (str): the location of the source csv file 
        chunk_size (int): the maximum number of tokens in each chunk
        chunk_overlap (int): how many tokens are shared between two subsequent chunks
        destination_folder (str): the location of the produced file 
    '''

    # Source file is required
    if source_file is None:
        return
    
    # Import the data that we exported from PubMed
    df_data = pd.read_csv(source_file)

    # Initialize langChain splitter to split abstracts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""])

    rows_list = []

    # Iterate over the data and chunk the abstracts (add an id number for each chunk)
    for index, row in tqdm(df_data.iterrows(), total=df_data.shape[0], desc='Chunking data'):    
        
        chunks = text_splitter.split_text(str(row['Abstract']))

        for i in range(len(chunks)):

            rows_list.append([row['PMID'], row['Title'], row['Abstract'], i, chunks[i], row['Key_words'], row['Authors'], row['Journal'], row['Year'], row['Month'], row['Source'], row['Country']])


    df_data_chunks = pd.DataFrame(rows_list, columns=["pmid", "title", "abstract", "chunk_id", "chunk", "key_words", "authors", "journal", "year", "month", "source", "country"])

    # Save the chunked data in a CSV file (We save the data in chunks so we can show the progress in tqdm)
    splits = np.array_split(df_data_chunks.index, 100)

    # The produced CSV file name will include the chunk size and the chunk overlap
    destination_file = destination_folder + f"data_chunks_{chunk_size}_{chunk_overlap}.csv"

    for index, split in enumerate(tqdm(splits, desc='Saving to CSV')):
        if index == 0:
            df_data_chunks.loc[split].to_csv(destination_file, mode='w', index=False)
        else:
            df_data_chunks.loc[split].to_csv(destination_file, header=None, mode='a', index=False)


if __name__ == "__main__":

    chunk_size = 500
    chunk_overlap = 100
       
    source_file = os.path.join(sys.path[0], "data\\exported_data.csv")
    destination_folder = os.path.join(sys.path[0], "data\\")

  
    if os.name != "nt":
        source_file = os.path.join(sys.path[0], "data/exported_data.csv")
        destination_folder = os.path.join(sys.path[0], "data/")

    
    chunk_data(source_file=source_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap, destination_folder=destination_folder)