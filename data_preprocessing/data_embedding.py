from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import numpy as np

# This script generate 

# Import the embedding model
model = SentenceTransformer('all-mpnet-base-v2')

# Read the chunked data
df_data = pd.read_csv('data_chunks.csv')

# Generate the embeddings for the chunked data
tqdm.pandas(desc='Generating embeddings')
df_data['Abstract_emb'] = df_data.progress_apply(lambda row: model.encode(row['Chunk']).tolist(), axis = 1)

# Save the chunked data with the embeddings in a CSV file (We save the data in chunks so we can show the progress in tqdm)
splits = np.array_split(df_data.index, 100)

for index, split in enumerate(tqdm(splits, desc='Saving to CSV')):
    if index == 0:
        df_data.loc[split].to_csv('data_chunks_emb.csv', mode='w', index=False)
    else:
        df_data.loc[split].to_csv('data_chunks_emb.csv', header=None, mode='a', index=False)