import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import csv


# Import chunked data
df = pd.read_csv('Project\\data\\exported_data_with_chunks.csv')

# Drop abstracts with NaN values 
df = df.dropna(subset=['Abstract'])

# Import the embedding model
model = SentenceTransformer('all-mpnet-base-v2')


# Embed the chunks between the start and end row (we do embeddings in batches as it task a lot of time)
start_row = 6000
end_row = 10000

rows_list = []

for index, row in tqdm(df.iterrows(), total=end_row, desc='Embedding data'):    
    
    if index < start_row:
        continue
    if index == end_row:
        break

    embedding = model.encode(row['Chunk']).tolist()

    rows_list.append([row['PMID'], row['Title'], row['Abstract'], row['Chunk_id'], row['Chunk'], embedding, row['Key_words'], row['Authors'], row['Journal'], row['Year'], row['Month'], row['Source'], row['Country']])


df_save = pd.DataFrame(rows_list, columns=["pmid", "title", "abstract", "chunk_id", "chunk", "embedding", "key_words", "authors", "journal", "year", "month", "source", "country"])


# Append the data to the existing embeddings file
# For the first batch use df_save.to_csv('Project\\data\\data_embeddings.csv',index=False)
df_save.to_csv('Project\\data\\data_embeddings.csv', mode='a', index=False, header=False)