from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils import llm_model, opensearch_vector_store, build_references, processed_output
from config import set_api_keys
from langchain.chains import RetrievalQA
from langchain import hub


# Setup all API tokens
set_api_keys()

# Initialize LLM model
llm = llm_model()

# Initialize OpenSearch vector and store and retriever
vector_store = opensearch_vector_store(index_name="pubmed_500_100")
retriever = vector_store.as_retriever(search_kwargs={"k": 3, "text_field":"chunk", "vector_field":"embedding"})

# Loads the latest version of RAG prompt
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


# Initialize FastAPI instance
app = FastAPI()

# Allow all origins during development
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    
    return {"message": output + "\n\n" + build_references(answer["source_documents"])}


@app.get("/retrieve_documents_sparse")
def retrieve_documents_sparse(message: str):
    response_message = f"FastAPI detected, that you said: {message}"
    return {"message": response_message}
