from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from utils import llm_model, opensearch_vector_store, build_references, processed_output
from config import set_api_keys
from langchain.chains import RetrievalQA
from langchain import hub

import asyncio

rag_pipeline = None
vectore_store = None
retriever = None
llm = None



# Initialize FastAPI instance
app = FastAPI()
INITIALIZING = False


SERVER_STATUS_MESSAGE = "Initializing FastAPI..."
SERVER_STATUS = "NOK"

# Allow all origins during development
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def initialization_task():
    initialize_rag_pipeline()

@app.get("/server_status")
def server_status(background_tasks: BackgroundTasks):

    global INITIALIZING
    if not INITIALIZING:
        INITIALIZING = True
        print("initializing...")
        background_tasks.add_task(initialization_task)

    print("***Server status: ", SERVER_STATUS, " - ", SERVER_STATUS_MESSAGE)
    return {"serverMessage": SERVER_STATUS_MESSAGE, "serverStatus": SERVER_STATUS}

def initialize_rag_pipeline():

    # Setup all API tokens
    global SERVER_STATUS_MESSAGE
    global SERVER_STATUS
    global rag_pipeline 
    global vectore_store 
    global retriever 
    global llm



    SERVER_STATUS_MESSAGE = "Setting up API keys..."
    SERVER_STATUS = "NOK"
    set_api_keys()

    # Initialize LLM model
    SERVER_STATUS_MESSAGE = "Initializing LLM model..."
    SERVER_STATUS = "NOK"
    llm = llm_model()


    # Initialize OpenSearch vector and store and retriever
    SERVER_STATUS_MESSAGE = "Initializing Opensearch backend..."
    SERVER_STATUS = "NOK"
    vector_store = opensearch_vector_store(index_name="pubmed_500_100")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3, "text_field":"chunk", "vector_field":"embedding"})

    # Loads the latest version of RAG prompt
    SERVER_STATUS_MESSAGE = "Setting up RAG pipeline..."
    SERVER_STATUS = "NOK"
    prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

    # Initialize langChain RAG pipeline
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose":"True"},
        verbose=True    
    )

    SERVER_STATUS_MESSAGE = "Setup finished!"
    SERVER_STATUS = "OK"



@app.get("/read_root")
def read_root(message: str):
    response_message = f"FastAPI detected, that you said: {message}"
    return {"message": response_message}


@app.get("/retrieve_documents_dense")
async def retrieve_documents(query_str: str):
    """
    A complete end-to-end RAG to answer user questions
    """
    answer = rag_pipeline.invoke({"query": query_str})    

    output = processed_output(answer["result"])
    
    return {"message": output + "_" + build_references(answer["source_documents"])}


@app.get("/retrieve_documents_sparse")
def retrieve_documents_sparse(message: str):
    response_message = f"FastAPI detected, that you said: {message}"
    return {"message": response_message}
