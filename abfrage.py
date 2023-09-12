"""
*** PCnewsGPT Abfrage - abfrage.py ***
    Änderung: V0.2.x ist eine komplette Neuadaption von V0.1, völlig OHNE langchain und mit optimiertem Abfragetext
"""

"""
Load Parameters, etc.
"""
from dotenv import load_dotenv
from os import environ as os_environ
load_dotenv()
persist_directory = os_environ.get('PERSIST_DIRECTORY','db')
collection_name = os_environ.get('COLLECTION_NAME','langchain')
embeddings_model_name = os_environ.get('EMBEDDINGS_MODEL_NAME','paraphrase-multilingual-mpnet-base-v2')
model_path = os_environ.get('MODEL_PATH','llama-2-13b-chat.ggmlv3.q4_0.bin')
model_n_ctx = int(os_environ.get('MODEL_N_CTX',2048))
model_temp = float(os_environ.get('MODEL_TEMP',0.4))
max_tokens = int(os_environ.get('MAX_TOKENS',500))
model_threads = int(os_environ.get('MODEL_THREADS',8))
model_n_gpu = int(os_environ.get('MODEL_GPU',1))
max_context_chunks = int(os_environ.get('MAX_CONTEXT_CHUNKS',4))
max_context_distance = float(os_environ.get('MAX_CONTEXT_DISTANCE',6.0))
# debugging
model_verbose = os_environ.get('MODEL_VERBOSE',"False") != "False"
hide_source = os_environ.get('HIDE_SOURCE',"False") != "False"
hide_source_details = os_environ.get('HIDE_SOURCE_DETAILS',"False") != "False"

"""
Initial banner Message
"""
print("\nPCnewsGPT Wissensabfrage V0.2.5.1\n")

"""
Initialize Chroma & Embeddings & Collections
"""
print(f"Datenbank {persist_directory}, Embeddings {embeddings_model_name} werden eingelesen...")
import chromadb
from chromadb.config import Settings
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet",
                                    persist_directory=persist_directory
                                ))
from chromadb.utils import embedding_functions
chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embeddings_model_name)
collection = chroma_client.get_collection(name=collection_name, embedding_function=chroma_ef)

"""
Initialize LLM
"""
print(f"KI-Model {model_path} wird geladen...")
from llama_cpp import Llama
llm = Llama(model_path=model_path,
            n_ctx=model_n_ctx,
            logits_all=False, 
            embedding = False,
            n_threads = model_threads,
            n_gpu_layers = model_n_gpu,
            verbose = model_verbose,
        )

# Central prompt template (note: substituted context has \n at end)
prompt_template = '###\nInformationen: {}' + \
                  'Anweisung: Beantworte die folgende Frage nur mit diesen Informationen.' + \
                  '\nFrage: {}\nAntwort: '

"""
main query loop - interactive questions and answers until empty line is entered
"""
while True:
    # get user question (empty line exits)
    question = input("\n### Frage: ")
    if question == "":
        break
    question = question.strip().rstrip('?').rstrip('.').strip() # increases response accuracy

    #create embeddings and ask DB for context
    result = collection.query(
        query_texts=[question],
        n_results=max_context_chunks
    )
    context_texts = result['documents'].__getitem__(0)
    context_sources = [dct['source'] for dct in result['metadatas'].__getitem__(0)] 
    context_distances = result['distances'].__getitem__(0)
    
    # generate LLM prompt
    context = ""
    for i,txt in enumerate(context_texts):
        # this assumes, that the context_text items came already cleaned up from the DB
        # concatenate items for prompt and ignore context information which is too far away
        if context_distances[i] <= max_context_distance:
            context = context + f"{i+1}. '{txt}'\n"


    # generate LLM prompt only if we have viable context
    if context == "":
        print(f"\n### Keine passenden Informationen in Wissensbasis gefunden.")
    else:
        prompt = prompt_template.format(context, question)    

        # print statistics at end of each answer together with sources
        if not hide_source:
            llm.verbose = True

        # stream output tokens
        print(f"\n### Antwort - bitte um etwas Geduld:")
        for chunk in llm(
            prompt = prompt,
            max_tokens = max_tokens,
            temperature = model_temp,
            echo=True,
            stop=["###"],
            stream = True,
            ):
            for choice in chunk['choices']:
                print(choice['text'], end='', flush=True)
        #print("",flush=True)
    
    # Print sources
    if not hide_source:
        print(f"\n### Quellen:")
        for i,source in enumerate(context_sources):
            print(f"[{i+1}] {source}, Distanz={context_distances[i]:.2f}", end="")
            if not (context_distances[i] <= max_context_distance):
                print(", *** ignoriert ***", end="")
            if not hide_source_details:
                # dont print real \n in source texts
                print(":\n'"+context_texts[i].replace("\n", "\\n")+"'")
            else:
                print("")
