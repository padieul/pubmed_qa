import pandas
import re, csv
from openai import OpenAI
import ast
import time
from opensearchpy import OpenSearch
from angle_emb import AnglE, Prompts
import os
from langchain import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Initialize AnglE embedding model
angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
# Enable Prompt.C for retrieval optimized embeddings
angle.set_prompt(prompt=Prompts.C)   

# OpenSearch instance parameters
host = 'localhost'
port = 9200
auth = ('admin', 'admin')

# Create the client with SSL/TLS enabled and disable warnings
client_OS= OpenSearch(
    hosts = [{'host': host, 'port': port}],    
    http_auth = auth,
    use_ssl = True,
    verify_certs = False,
    ssl_show_warn = False,
)

client_OpenAI = OpenAI(
    api_key = "" # OpenAI API TOKEN TO BE INSERTED HERE
)


os.environ["HUGGINGFACEHUB_API_TOKEN"] = "" # HUGGINGFACEHUB API TOKEN TO BE INSERTED HERE

repo_id = "tiiuae/falcon-7b-instruct" 

falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1}
)


def create_test_csv(file_path):
    with open(file_path, 'w', newline='') as file:
        # Create a CSV writer object with tab as the delimiter
        csv_writer = csv.writer(file, delimiter='\t')  # Specify tab as the delimiter

        header = ["pmid", "pmid2", "pmid3", "chunk_id", "chunk_id2", "chunk_id3", "chunk", "chunk2", "chunk3",
        "question_type", "question", "answer", "similarity_search", "keywords_if_complex_and_sparse", "generator_model", " warning_while_generation"]
        csv_writer.writerow(header)


def get_useful_records(original_file_path):
    '''
    This function is used to get the useful data records that we can use to sample from.
    It takes the original file path and takes the data records, and creates a new file with processed data
    that has keywords - and the number of keywords should be 4 to 6.
    This function is used once, so that we do not need to preprocess the documents multiple times
    each time we want to generate questions
    '''
    df_processed = pandas.read_csv(original_file_path, usecols=['pmid', 'title', 'chunk_id', 'chunk', 'embedding', 'key_words'])

    # Here we need to keep the data records with keywords because we need the keywords
    # for our similarity search - to find related documents to generate complex questions
    df_processed = df_processed[df_processed['key_words'].notna()] # data records with key_words

    df_processed['key_words'] = df_processed['key_words'].apply(ast.literal_eval) # convert the key_words to list type

    # After investigation of the data records, we observed
    # that the more keywords a data record has, more generic keywords are
    # and we also need to have date records with at least 3 key_words for our similarity search
    # so we decided on the size: minimum: 4 keywords and maximum: 6 keywords
    df_processed = df_processed[df_processed['key_words'].apply(lambda x: isinstance(x, list) and len(x) >= 4 and len(x) <= 6)]

    df_processed.to_csv("data_preprocessing/data/processed_data_for_testing_500_100.csv", index=False)

    return


def get_similar_chunks(result_of_similarity_search, pmid_original, chunk_id_original):
    '''
    This function is used to get the attributes of the similar chunks.
    It takes the result of a similarity search and,  
    the pmid and the chunk id of the chunk whose similar chunks should be returned.

    It returns three lists that are;
    1) list of similar chunks, 2) list of pmid's of those chunks, 3) chunk id's of those chunks
    '''
    similar_chunks = [] 
    similar_chunks_pmids = []
    similar_chunks_chunk_ids = []
    for hit in result_of_similarity_search['hits']['hits']:
        # id = hit['_id']
        # score = hit['_score']
        pmid_similar = hit['_source']['pmid']
        chunk_id_similar = hit['_source']['chunk_id']  
        if pmid_similar == pmid_original and chunk_id_similar == chunk_id_original: # if the found similar chunk is the chunk itself
            # print("FOUND ITSELF") # this is for debugging purposes
            continue
                    
        chunk_similar = hit['_source']['chunk']

        similar_chunks.append(chunk_similar)
        similar_chunks_pmids.append(pmid_similar)
        similar_chunks_chunk_ids.append(chunk_id_similar)
    return similar_chunks, similar_chunks_pmids, similar_chunks_chunk_ids

def get_attributes_for_most_similar(similar_chunks_pmids, similar_chunks_chunk_ids, similar_chunks):
    '''
    This function takes the pmid's, chunk id's and chunks of the most similar chunks
    It is used to assign the values for the attributes of the most similar chunk.
    Those attributes are pmid, chunk id and chunk.
    This function is only used in the case of taking the most similar chunk
    '''

    pmid2 = similar_chunks_pmids[0]
    chunk_id2 = similar_chunks_chunk_ids[0]
    chunk2 = similar_chunks[0]
    return pmid2, chunk_id2, chunk2


def get_attributes_for_two_most_similar(similar_chunks_pmids, similar_chunks_chunk_ids, similar_chunks):
    '''
    This function takes the pmid's, chunk id's and chunks of the two most similar chunks
    It is used to assign the values for the attributes of the most similar chunk.
    Those attributes are pmid's, chunk id's and chunks.
    This function is only used in the case of taking the most two similar chunk
    '''

    pmid2 = similar_chunks_pmids[0]
    pmid3 = similar_chunks_pmids[1]
    chunk_id2 = similar_chunks_chunk_ids[0]
    chunk_id3 = similar_chunks_chunk_ids[1]
    chunk2 = similar_chunks[0]
    chunk3 = similar_chunks[1]

    return pmid2, pmid3, chunk_id2, chunk_id3, chunk2, chunk3


def gpt_3_5_turbo(prompt):
    '''

    '''
    time.sleep(30) # sleep each time before sending any prompt to gpt
    chat_completion = client_OpenAI.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-3.5-turbo-1106"
    )
    reply = chat_completion.choices[0].message.content
    return reply


def falcon_7b_instruct(prompt):
    template = """{prompt}"""

    prompt_to_falcon = PromptTemplate.from_template(template)

    llm_chain = LLMChain(prompt=prompt_to_falcon, llm=falcon_llm)

    reply = llm_chain.run(prompt)
    return reply


def write_to_test_set(pmid, pmid2, pmid3, 
                      chunk_id, chunk_id2, chunk_id3,
                      chunk, chunk2, chunk3,
                      question_type, reply, similarity_search, keywords_if_complex_and_sparse, generator_model):
    '''
    This function is used to check the validity of the reply by the generator model,
        if necessary change the format of the reply,
        and write the new record to the testing set
    '''
    # pmid pmid2 pmid3 chunk_id chunk_id2 chunk_id3 chunk chunk2 chunk3 question_type question answer 
    # similarity_search keywords_if_complex_and_sparse generator_model warning_while_generation

    if reply.lower() == "na" or ("na" in reply.lower() and len(reply) < 10):
        warning_while_generation = f"WARNING: GENERATED TEXT IS 'N/A'\n \
Original Reply: '{reply}'\nPMID:{pmid}, CHUNK ID: {chunk_id}, Question Type: {question_type}\n\n"
            
        # writing the warning to a txt file
        with open("data_preprocessing/test_data/warnings.txt", 'a') as file:
            file.write(warning_while_generation)
        return
    try:
        # Check if the model gave the response in a correct format
        
        reply_list = ast.literal_eval(reply)
        if isinstance(reply_list, list) and all(isinstance(item, str) for item in reply_list):
            # everything is good, we can add this to our dataset
            warning_while_generation = "N/A"
            test_set_file_path = 'data_preprocessing/test_data/test_dataset.csv'

            with open(test_set_file_path, 'a', newline='') as file:
                csv_writer = csv.writer(file, delimiter='\t')
                        
                new_record = [pmid, pmid2, pmid3, chunk_id, chunk_id2, chunk_id3, chunk, chunk2, chunk3, 
                              question_type] + reply_list + [similarity_search, keywords_if_complex_and_sparse, generator_model, warning_while_generation]
                
                csv_writer.writerow(new_record)
                        
        else:
            warning_while_generation = f"WARNING: GENERATION IS NOT IN THE CORRECT FORMAT - LIST ELEMENTS ARE NOT STRINGS\n \
THIS IS A RARE CASE THAT CURRENTLY HAS NO SOLUTION\nPMID:'{pmid}', CHUNK ID: '{chunk_id}', Question Type: '{question_type}'\n\n"

            # writing the warning to a txt file
            with open("data_preprocessing/test_data/warnings.txt", 'a') as file:
                file.write(warning_while_generation)
            
    except (SyntaxError, ValueError):        
        warning_while_generation = f"WARNING: A LIST IS NOT GENERATED! - REFORMATTED TO A LIST FORMAT [MAY NOT BE ACCURATE REFORMMATING]"
        
        reply = "".join(reply) # some 
        question_start = reply.lower().find("question:")
        answer_start = reply.find("answer:")

        question = reply[question_start + len("question:"):answer_start].strip()
        answer = reply[answer_start + len("answer:"):].strip()

        reformatted_reply = [question, answer] # reformatted reply

        test_set_file_path = 'data_preprocessing/test_data/test_dataset.csv'

        with open(test_set_file_path, 'a', newline='') as file:
            csv_writer = csv.writer(file, delimiter='\t')
                        
            new_record = [pmid, pmid2, pmid3, chunk_id, chunk_id2, chunk_id3, chunk, chunk2, chunk3, 
                              question_type] + reformatted_reply + [similarity_search, keywords_if_complex_and_sparse, generator_model, warning_while_generation]
                
            csv_writer.writerow(new_record)

        warning_while_generation += f"\nOriginal Reply: '{reply}'\nReformatted Reply: '{reformatted_reply}'\nPMID:{pmid}, CHUNK ID: {chunk_id}, Question Type: {question_type}\n\n"
        with open("data_preprocessing/test_data/warnings.txt", 'a') as file:
                file.write(warning_while_generation)

            
    return

# create_test_csv("data_preprocessing/test_data/test_dataset.csv") # To create the test set csv file - run only once

# run only once to process the original data embeddings file and create a new one
# get_useful_records("data_preprocessing/data/data_embeddings_500_100.csv") 

# Here we store the chunks that we have already processed, in order not to use in the future
df_already_processed_documents = pandas.read_csv("data_preprocessing/test_data/test_dataset.csv", usecols=["pmid", "chunk_id"], sep='\t')
already_processed_documents = set(df_already_processed_documents['pmid'].astype(str) + '_' + df_already_processed_documents['chunk_id'].astype(str))
# print(already_processed_documents)

df_data_embeddings = pandas.read_csv("data_preprocessing/data/processed_data_for_testing_500_100.csv", usecols=['pmid', 'title', 'chunk_id', 'chunk', 'embedding', 'key_words'])
# print(df_data_embeddings.head())
# print(df_data_embeddings.shape)

# Here we randomly take 100 data records to generate questions
sampled_data_records = df_data_embeddings.sample(n=100)
# print(sampled_data_records)


# Here we randomly take 40 records from 100 selected
# With these 40 chunks, only one keyword will be used to find similar chunks
# and eventually to generate a complex question
one_keyword_records = sampled_data_records.sample(n=40)

# Dropping 40 records (that uses one keyword for similarity) from the original 100 selected records.
# sampled_data_records = sampled_data_records.drop(one_keyword_records.index)

# Here we randomly take another 40 records from the remaining 60, initially selected records.
# these chunks/records will use 2 keywords to find similar chunks to generate complex questions
two_keywords_records = sampled_data_records.sample(n=40)

# Dropping the records that will use 2 keywords for similariy search
# and getting the final 20 records which will use 3 keywords to find related chunks/records for complex question generation
three_keywords_records = sampled_data_records.drop(two_keywords_records.index)


# Now we have 3 dataframes with the sizes of 40, 40, 20 respectively
# Now we can generate the questions and do the complex question generation differently
# depending on the dataframe that our record/chunk belongs to

# 30 of 40 chunks that uses one keyword to find related chunks
# will be compined with one another chunk that is the most similar to it for complex question generation
# the oter 10 chunks will be compined with 2 other chunks that are the most similar to it for complex question generation
count_for_one_keyword_two_chunks = 0 # 30

# again 30 of 40 chunks that use two keywords to find related chunks
# will be compined with one another chunk, and 10 chunks will be combined with 2 other most similar chunks
# for complex question generation
count_for_two_keywords_two_chunks = 0 # 30

# 15 of 20 chunks that use three keywords to find related chunks
# will be compined with one another chunk, and 5 chunks will be combined with 2 other most similar chunks
count_for_three_keywords_two_chunks = 0 # 15


question_types = ["Confirmation", "Factoid-type", "List-type", "Causal", "Hypothetical", "Complex"] # 6 types


prompts = ["You to generate a Yes/No question that require an understanding of a given context and deciding a \
boolean value for an answer, e.g., 'Is Paris the capital of France?'. ",
            "You need to generate a Factoid-type Question [what, which, when, who, how]: These usually begin with a “wh”-word. \
An answer then is commonly short and formulated as a single sentence. e.g., 'What is the capital of France?'. ",
            "You need to generate a List-type Question: The answer is a list of items, e.g.,'Which cities have served as the \
capital of France throughout its history?'. ",
            "You need to generate a Causal Questions [why or how]: Causal questions seek reasons, explanations, and \
elaborations on particular objects or events, e.g., “Why did Paris become the capital of France?” \
Causal questions have descriptive answers that can range from a few sentences to whole paragraphs.",
            "You need to generate a Hypothetical Question: These questions describe a hypothetical scenario \
and usually start with “what would happen if”, e.g., 'What would happen if Paris airport closes for a day?'.",
            "You need to generate a Complex Question: Complex questions require multi-part reasoning by understanding \
the semantics of multiple text snippets, e.g. 'What cultural and historical factors contributed to the development of the \
Louvre Museum as a world-renowned art institution?' which requires inferring information from multiple documents to generate an answer."
]

# print(prompts)

# Here I concatenate all the data records but we know that 
# first we have chunks that we will use one keyword to find its similar chunks,
# then we have records that we will use two keywords to find its similar chunks,
# and finally we have records that we will use three keywords to find its similar chunks.
# Rows of these records are stacked on top of the other.
all_records = pandas.concat([one_keyword_records, two_keywords_records, three_keywords_records], ignore_index=True)

num_of_records = all_records.shape[0] # the number of samples we took 
for i in range(num_of_records):
    pmid, title, chunk_id, chunk, embedding, key_words = all_records.iloc[i, ]
    key_words = ast.literal_eval(key_words)

    # checking if we have already processed this chunk
    chunk_identifier = str(pmid) + '_' + str(chunk_id)
    if chunk_identifier not in already_processed_documents:
        for j in range(6): # for each chunk we try to generate all 6 types of questions
            question_type = question_types[j]

            # COMPLEX QUESTIONS
            if question_type == "Complex": # Complex question
                # Here we find how many keywords we need to use to find similar chunks
                # and also get the keywords to be used
                
                # BEFORE GENERATING THE QUESTION, WE NEED TO DECIDE ON SOME PARAMETERS;
                # NUMBER OF KEYWORDS for sparse search
                # NUMBER OF SIMILAR CHUNKS that we are looking for

                # ONE KEYWORD SEARCH
                if i < len(one_keyword_records): # STILL NOT DONE WITH ONE KEYWORD SEARCH
                    keywords = key_words[0] # GETTING ONLY ONE KEYWORD FOR OUR SEARCH
                    if count_for_one_keyword_two_chunks < 30: 
                        # ONLY THE MOST SIMILAR CHUNK
                        # because we agreed on finding one similar chunk for 30 of 40 records
                        num_of_similar_chunks = 1
                        count_for_one_keyword_two_chunks += 1
                    else:
                        # otherwise we find TWO MOST SIMILAR CHUNKS
                        num_of_similar_chunks = 2

                # TWO KEYWORDS SEARCH
                elif i < len(one_keyword_records) + len(two_keywords_records): # STILL NOT DONE WITH TWO KEYWORDS SEARCH
                    keywords = key_words[0] + " " + key_words[1] # creating the QUERY OF KEYWORDS for our search
                    if count_for_two_keywords_two_chunks < 30: 
                        # ONLY THE MOST SIMILAR CHUNK
                        # because we agreed on finding one similar chunk for 30 of 40 records
                        num_of_similar_chunks = 1
                        count_for_two_keywords_two_chunks += 1
                    else:
                        # TWO MOST SIMILAR CHUNKS
                        num_of_similar_chunks = 2

                # THREE KEYWORDS SEARCH
                else:
                    # THREE KEYWORDS SEARCH
                    keywords = key_words[0] + " " + key_words[1] + " " + key_words[2] # creating the QUERY OF KEYWORDS for our search
                    if count_for_three_keywords_two_chunks < 15:
                        # ONLY THE MOST SIMILAR CHUNK
                        # because we agreed on finding one similar chunk for 15 of 20 records
                        num_of_similar_chunks = 1
                        count_for_three_keywords_two_chunks += 1
                    else:
                        # TWO MOST SIMILAR CHUNKS
                        num_of_similar_chunks = 2
                
                # print(keywords)

                # NOW WE HAVE EVERYTHING TO FIND THE MOST SIMILAR CHUNK(S)
                # WE HAVE THE KEYWORDS AND WE HAVE SIZE OF OUR SEARCH, MEANING HOW MANY SIMILAR CHUNKS ARE WE INTERESTED IN
                
                # WE NEED TO EXTRACT THE CHUNKS HERE
                        
                # SPARSE SEARCH
                query_sparse = keywords 
                size = num_of_similar_chunks + 1 # + 1 is because of the fact that the chunk itself may appear as one of the similar chunks
                search_query_sparse = {
                    "query": {
                        "match": {
                            "chunk": query_sparse
                        }
                    },
                    "size": size
                }

                results_sparse = client_OS.search(index="pubmed_500_100", body=search_query_sparse)

                # DENSE SEARCH
                # getting embedding of the query for dense search
                query_embedding_dense = angle.encode({'text': chunk}) # maybe we can use the embedding as probably it's computed ??

                search_query_dense = {    
                    "query": {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding_dense[0].tolist(),
                                "k": size # is this the number of similar chunks that i need to find??
                            }
                        }
                    }
                }

                results_dense = client_OS.search(index="pubmed_500_100", body=search_query_dense)

                similar_chunks_sparse, similar_chunks_pmids_sparse, similar_chunks_chunk_ids_sparse = get_similar_chunks(results_sparse, pmid, chunk_id)
                similar_chunks_dense, similar_chunks_pmids_dense, similar_chunks_chunk_ids_dense = get_similar_chunks(results_dense, pmid, chunk_id)

                
                if num_of_similar_chunks == 1: # LOOKING FOR THE MOST SIMILAR CHUNK
                    if len(similar_chunks_sparse) > 0: # THE MOST SIMILAR FROM SPARSE IF AVAILABLE
                        pmid2_sparse, chunk_id2_sparse, chunk2_sparse = get_attributes_for_most_similar(similar_chunks_pmids_sparse, similar_chunks_chunk_ids_sparse, similar_chunks_sparse)
                        
                        prompt_sparse = prompts[j] + "You need to use the given 2 different given text snippets to generate the question!!. You also need to generate an answer for your question. \
The first given text snippet is: " + chunk + " The second given text snippet is: " + chunk2_sparse + " Remember and be careful: each of the entries in the lists should be a string with quotation marks!! " + "You \
just give a python list of size 2 with question and its answer for the given chunk at the end. That is like ['a question', 'an answer to that question']. \
IT IS SOO IMPORTANT TO GIVE ME A LIST OF 2 STRINGS THAT IS QUESTION AND ANSWER. IF YOU THING THAT THIS KIND OF QUESTION CANNOT BE GENERATED JUST TELL ME 'NA'.\
IF YOU ALSO THING THAT GENERATING A QUESTION FROM THESE 2 GIVEN TEXT SNIPPETS DOES NOT MAKE SENSE JUST TELL ME 'NA' AGAIN!! DO NOT HALLUSINATE!!!"


                    if len(similar_chunks_dense) > 0: # THE MOST SIMILAR FROM DENSE IF AVAILABLE
                        pmid2_dense, chunk_id2_dense, chunk2_dense = get_attributes_for_most_similar(similar_chunks_pmids_dense, similar_chunks_chunk_ids_dense, similar_chunks_dense)
                        
                        prompt_dense = prompts[j] + "You need to use the given 2 given different given text snippets to generate the question!!. You also need to generate an answer for your question. \
The first given text snippet is: " + chunk + " The second given text snippet is: " + chunk2_dense + " Remember and be careful: each of the entries in the lists should be a string with quotation marks!! " + "You \
just give a python list of size 2 with question and its answer for the given chunk at the end. That is like ['a question', 'an answer to that question']. \
IT IS SOO IMPORTANT TO GIVE ME A LIST OF 2 STRINGS THAT IS QUESTION AND ANSWER. IF YOU THING THAT THIS KIND OF QUESTION CANNOT BE GENERATED JUST TELL ME 'NA'.\
IF YOU ALSO THING THAT GENERATING A QUESTION FROM THESE 2 GIVEN TEXT SNIPPETS DOES NOT MAKE SENSE JUST TELL ME 'NA' AGAIN!! DO NOT HALLUSINATE!!!"

                elif num_of_similar_chunks == 2: # LOOKING FOR THE MOST TWO SIMILAR CHUNKS
                    if len(similar_chunks_sparse) > 1:
                        pmid2_sparse, pmid3_sparse, chunk_id2_sparse, chunk_id3_sparse, chunk2_sparse, chunk3_sparse = get_attributes_for_two_most_similar(similar_chunks_pmids_sparse, similar_chunks_chunk_ids_sparse, similar_chunks_sparse)
                        
                        prompt_sparse = prompts[j] + "You need to use the given 3 different given text snippets to generate the question!!. You also need to generate an answer for your question. \
The first giventext snippet is: " + chunk + " The second given text snippet is: " + chunk2_sparse + " The third given text snippet is: " + chunk3_sparse + " Remember and be careful: each of the entries in the lists should be a string with quotation marks!! " + "You \
just give a python list of size 2 with question and its answer for the given chunk at the end. That is like ['a question', 'an answer to that question']. \
IT IS SOO IMPORTANT TO GIVE ME A LIST OF 2 STRINGS THAT IS QUESTION AND ANSWER. IF YOU THING THAT THIS KIND OF QUESTION CANNOT BE GENERATED JUST TELL ME 'NA'.\
IF YOU ALSO THING THAT GENERATING A QUESTION FROM THESE 2 GIVEN TEXT SNIPPETS DOES NOT MAKE SENSE JUST TELL ME 'NA' AGAIN!! DO NOT HALLUSINATE!!!"

                    if len(similar_chunks_dense) > 1:
                        pmid2_dense, pmid3_dense, chunk_id2_dense, chunk_id3_dense, chunk2_dense, chunk3_dense = get_attributes_for_two_most_similar(similar_chunks_pmids_dense, similar_chunks_chunk_ids_dense, similar_chunks_dense)
                        
                        # MADE SOME CHANGES TO THIS PROMPT - CONSIDER THISS!!!!!!
                        prompt_dense = prompts[j] + "You need to use the 3 different given text snippets to generate the question!!. You also need to generate an answer for your question. \
The first given text snippet is: " + chunk + " The second given text snippet is: " + chunk2_dense + " The third given text snippet is: " + chunk3_dense + " Remember and be careful: each of the entries in the lists should be a string with quotation marks!! " + "You \
just give a python list of size 2 with question and its answer for the given chunk at the end. That is like ['a question', 'an answer to that question']. \
IT IS SOO IMPORTANT TO GIVE ME A LIST OF 2 STRINGS THAT IS QUESTION AND ANSWER."

                        # use the model to generate question and answer pair
                        reply_gpt = gpt_3_5_turbo(prompt_dense)

                        # write the generated record to the test set
                        write_to_test_set(pmid, pmid2_dense, pmid3_dense, 
                                          chunk_id, chunk_id2_dense, chunk_id3_dense,
                                          chunk, chunk2_dense, chunk3_dense,
                                          question_type, reply_gpt, "Dense", "N/A", "gpt-3.5-turbo-1106")


#             else:
#                 prompt = prompts[j] + "You need to use the given text snippet to generate the question!!. You also need to generate an answer for your question. \
# The given text snippet is: " + chunk + " Remember and be careful: each of the entries in the lists should be a string with quotation marks!! " + "You \
# just give a python list of size 2 with question and its answer for the given chunk at the end. That is like ['a question', 'an answer to that question']. \
# IT IS SOO IMPORTANT TO GIVE ME A LIST OF 2 STRINGS THAT IS QUESTION AND ANSWER. IF YOU THING THAT THIS KIND OF QUESTION CANNOT BE GENERATED JUST TELL ME 'NA'.\
# DO NOT HALLUSINATE!!!"

#                 # use the model to generate question and answer pair
#                 reply_gpt = gpt_3_5_turbo(prompt)

#                 # write the generated record to the test set
#                 write_to_test_set(pmid, "N/A", "N/A", 
#                       chunk_id, "N/A", "N/A",
#                       chunk, "N/A", "N/A",
#                       question_type, reply_gpt, "N/A", "N/A", "gpt-3.5-turbo-1106")

            # print(question_type)
            # print(prompt)
            # reply_gpt = gpt_3_5_turbo(prompt)
                
        already_processed_documents.add(chunk_identifier)