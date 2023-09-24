"""
*** PCnewsGPT Abfrage - abfrage.py ***

    Änderungen:
    V0.2.6.x - Einbeziehen von Datum in Quellen
    V0.2.5.x - Opimieren question-Vorverarbeitung, mehr Parameter, ignorieren großer chromaDB Distanzen, Schätzung Initialantwortzeit
    V0.2.x -komplette Neuadaption von V0.1, völlig OHNE langchain und mit optimiertem Abfragetext
    V0.1.x - pure langchain basierte Adaption von privateGPT, detuscher Prompt
"""

"""
Initial banner Message
"""
print("\nPCnewsGPT Wissensabfrage V0.2.6\n")

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
model_prompt_per_s = int(os_environ.get('MODEL_PROMPT_PER_S',40))
max_context_chunks = int(os_environ.get('MAX_CONTEXT_CHUNKS',4))
max_context_distance = float(os_environ.get('MAX_CONTEXT_DISTANCE',6.0))
# debugging
hide_source = os_environ.get('HIDE_SOURCE',"False") != "False"
hide_source_details = os_environ.get('HIDE_SOURCE_DETAILS',"False") != "False"

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
            verbose = False,                # quiet on initial load
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

    # increase response accuracy by eliminating trailing ? or . or surrounding spaces
    question = question.strip().rstrip('?').rstrip('.').strip() 
    
    #create embeddings and ask DB for context
    result = collection.query(
        query_texts=[question],
        n_results=max_context_chunks
    )
    context_texts = result['documents'].__getitem__(0)
    context_distances = result['distances'].__getitem__(0)
    context_metadatas = result['metadatas'].__getitem__(0)
    # re-format context_date (D:YYYYMMMDDD... -> YYYY-MM-DD)
    context_dates = []
    for i,metadata in enumerate(context_metadatas):
        date = metadata.get('creationDate','')
        context_dates.append(f'{date[2:6]}-{date[6:8]}-{date[8:10]}' if date.lower().startswith('d:') else date)
    
    # generate LLM prompt context
    context = ""
    for i,txt in enumerate(context_texts):
        # this assumes, that the context_text items came already cleaned up from the DB
        # concatenate items for prompt and ignore context information which is too far away
        if context_distances[i] <= max_context_distance:
            context += f'{i+1}. '
            if context_dates[i] != "":
                context += f'von Datum {context_dates[i]}:'
            context += f"'{txt}'\n"

    # generate LLM prompt only if we have viable prompt context
    if context == "":
        print(f"\n### Keine passenden Informationen in Wissensbasis gefunden.")
    else:
        prompt = prompt_template.format(context, question)    

        # determine number of tokens in prompt for answer-delay estimation
        tokens=llm.tokenize(prompt.encode('utf-8'),add_bos=False)
        prompt_eval_time_estimate = int(round(len(tokens) / model_prompt_per_s,0))

        # print statistics at end of each answer together with sources
        if not hide_source:
            llm.verbose = True

        # stream output tokens
        print(f"\n### Antwort (bitte initial um ca. {prompt_eval_time_estimate}s Geduld):")
        for chunk in llm(
            prompt = prompt,
            max_tokens = max_tokens,
            temperature = model_temp,
            echo=False,
            stop=["###"],
            stream = True,
            ):
            for choice in chunk['choices']:
                print(choice['text'], end='', flush=True)
        # Model automatically prints statistics after answer if not hide_source
    
    # Print sources
    if not hide_source:
        print(f"\n### Quellen:")
        for i,metadata in enumerate(context_metadatas):
            if (context_distances[i] <= max_context_distance):
                print(f"[{i+1}] {metadata.get('source','').split('/')[-1]} Seite:{metadata.get('page','-')} Textteil:{metadata.get('chunk','-')} Distanz:{context_distances[i]:.2f} Datum:{context_dates[i]}", end="")
                if not hide_source_details:
                # dont print real \n in source texts
                    print(" :\n'"+context_texts[i].replace("\n", "\\n")+"'")
                else:
                    print("")
            else:
                print(f"[{i+1}] Ignoriert:{metadata.get('source')} Seite:{metadata.get('page','-')} Textteil:{metadata.get('chunk','-')} Distanz:{context_distances[i]:.2f}")
