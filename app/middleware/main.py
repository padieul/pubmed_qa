from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain import hub

from pydantic import BaseModel, Field
from typing import List, Optional

from utils import llm_model, opensearch_vector_store, build_references, processed_output
from config import set_api_keys
from models import VariableRetriever, RetrievalFilter

rag_pipeline = None
vectore_store = None
retriever = None
prompt = None
llm = None

filtering = False



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

    # Define as global variables so that 
    # they can be accessed by server_status
    global SERVER_STATUS_MESSAGE
    global SERVER_STATUS
    global rag_pipeline 
    global vectore_store 
    global retriever 
    global prompt
    global llm

    # Setup all API tokens
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
    retriever = vector_store.as_retriever(search_kwargs={"k": 20, "text_field":"chunk", "vector_field":"embedding"})
    default_filter = {"title": "", "years": [], "keywords": []}
    print(default_filter)
    default_retriever = VariableRetriever(vectorstore=retriever, retrieval_filter=RetrievalFilter(default_filter))

    # Loads the latest version of RAG prompt
    SERVER_STATUS_MESSAGE = "Setting up RAG pipeline..."
    SERVER_STATUS = "NOK"
    prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")

    # Initialize langChain RAG pipeline
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=default_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose":"True"},
        verbose=True    
    )

    SERVER_STATUS_MESSAGE = "Setup finished!"
    SERVER_STATUS = "OK"

def reinitialize_rag_pipeline_retriever(filter_dict: dict):

    global SERVER_STATUS_MESSAGE
    global SERVER_STATUS
    global rag_pipeline
    
    filter_retriever = VariableRetriever(vectorstore=retriever, retrieval_filter=RetrievalFilter(filter_dict))

    # Loads the latest version of RAG prompt
    SERVER_STATUS_MESSAGE = "Setting up RAG pipeline..."
    SERVER_STATUS = "NOK"

    # Initialize langChain RAG pipeline
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=filter_retriever,
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


class Filter(BaseModel):
    title: Optional[str] = Field(None, description="The title to filter by")
    year_range: Optional[List[str]] = Field(None, description="The range of years to filter by")
    keywords: Optional[List[str]] = Field(None, description="The keywords to filter by")

class RequestBody(BaseModel):
    filter: Filter
    query_str: str

@app.post("/retrieve_documents_dense_f")
async def retrieve_documents(body: RequestBody):
    """
    A complete end-to-end RAG to answer user questions
    """
    global filtering

    filter = body.filter
    query_str = body.query_str
    filter_data = {
        "title": str(filter.title) if filter.title else "",
        "years": [str(year) for year in filter.year_range] if filter.year_range else [],
        "keywords": [str(keyword) for keyword in filter.keywords] if filter.keywords else []
        }
   
    # all values are empty -> no filtering
    if all(not value for value in filter_data.values()):
        print("---FILTERING: EVERYTHING IS EMPTY ---")
        if not filtering:
            print("---FILTERING: ALREADY DEFAULT ---")
            # no need to reinitialize the retriever because 
            # it is already the default retriever
            answer = rag_pipeline.invoke({"query": query_str}) 
        else:
            print("---FILTERING: SETTING TO DEFAULT ---")
            default_filter = {"title": "", "years": [], "keywords": []}
            reinitialize_rag_pipeline_retriever(default_filter)
            answer = rag_pipeline.invoke({"query": query_str}) 
            filtering = False
    else:
        print("---FILTERING: SETTING TO CUSTOM ---")
        reinitialize_rag_pipeline_retriever(filter_data)
        answer = rag_pipeline.invoke({"query": query_str}) 
        filtering = True

    output = processed_output(answer["result"])
    return {"message": output + "_" + build_references(answer["source_documents"])}




@app.get("/retrieve_documents_dense")
async def retrieve_documents(query_str: str):
    """
    OLD implementation: A complete end-to-end RAG to answer user questions
    """
    answer = rag_pipeline.invoke({"query": query_str})  
    output = processed_output(answer["result"])
    
    return {"message": output + "_" + build_references(answer["source_documents"])}


@app.get("/retrieve_documents_sparse")
def retrieve_documents_sparse(message: str):
    response_message = f"FastAPI detected, that you said: {message}"
    return {"message": response_message}
