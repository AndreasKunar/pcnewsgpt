"""
*** PCnewsGPT Abfrage - abfrage.py ***

    + german language queries
"""

"""
Load Parameters, etc.
"""
from dotenv import load_dotenv
from os import environ as os_environ, path as os_path
load_dotenv()
persist_directory = os_environ.get('PERSIST_DIRECTORY','db')
embeddings_model_name = os_environ.get('EMBEDDINGS_MODEL_NAME','paraphrase-multilingual-mpnet-base-v2')
model_path = os_environ.get('MODEL_PATH','llama-2-13b-chat.ggmlv3.q4_0.bin')
model_n_ctx = os_environ.get('MODEL_N_CTX',2048)
model_temp = os_environ.get('MODEL_TEMP',0.4)
target_source_chunks = int(os_environ.get('TARGET_SOURCE_CHUNKS',4))
mute_stream = os_environ.get('MUTE_STREAM',"False") != "False"
# debugging
hide_source = os_environ.get('HIDE_SOURCE',"False") != "False"
hide_source_details = os_environ.get('HIDE_SOURCE_DETAILS',"False") != "False"
#langchain & LLM debugging
from langchain import debug as langchain_debug
if os_environ.get('LANGCHAIN_DEBUG',"False") != "False":
    langchain_debug=True

"""
Initial banner Message
"""
print("PCnewsGPT Wissensabfrage V0.1\n")

"""
Initialize Embeddings
"""
from langchain.embeddings import HuggingFaceEmbeddings
print(f"Embeddings {embeddings_model_name} werden eingelesen...")
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

"""
Initialize ChromaDB
"""
from langchain.vectorstores import Chroma
from chromadb.config import Settings as Chroma_Settings
print(f"Wissensdatenbank in {persist_directory} wird geöffnet...")
chroma_settings = Chroma_Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=chroma_settings)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

"""
Initialize the LLM
"""
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import PromptTemplate
from langchain.llms import LlamaCpp

llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=[StreamingStdOutCallbackHandler()], 
            temperature=model_temp, n_threads=8, n_gpu_layers=0, verbose=False) # type: ignore
# tweak - set verbose_False in underlying LlamaCpp.py to surpress llama_print_timings
llm.client.verbose = False        

# Prepare prompt
prompt_template = """Verwenden Sie die folgenden Kontextinformationen, um die Frage am Ende zu beantworten. Wenn Sie die Antwort nicht kennen, sagen Sie einfach, dass Sie es nicht wissen, versuchen Sie nicht, eine Antwort zu erfinden.

{context}

Frage: {question}
Antwort:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# langchain orchestration
chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=not hide_source,
    chain_type_kwargs=chain_type_kwargs,
)

"""
Interactive questions and answers until empty line is entered
"""
from regex import sub as regex_sub

while True:
    query = input("\n### Frage: ")
    if query == "":
        break

    # Get the answer from the chain and stream-print the result
    print(f"\n### Antwort mittels KI-Modell {model_path}, temperature: {model_temp} - bitte um etwas Geduld:")

    res = qa(query)
    answer, docs = res['result'], [] if hide_source else res['source_documents']

    # Print the relevant sources used for the answer as processable data
    print("",flush=True)
    if docs:
        print("\n### Quellen:")
        for i,document in enumerate(docs):
            print(f"[{i}] 'source':'{document.metadata['source']}'", end="")
            if not hide_source_details:
                # remove all \n from text
                txt = document.page_content.replace('\n', '')
                # replace single quotes with double quotes
                txt = txt.replace("'", '"')
                print(f",'page_content':'{txt}'", end="")
                if i < len(docs)-1:
                    print(",")
            else:
                print()
