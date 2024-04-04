"""
*** PCnewsGPT Abfrage - abfrage.py ***

    Änderungen:
    V0.2.9 - Prompt-Optimierungen
    V0.2.8 - Optimierungen für Mistral-AI's "tiny" Mistral-7B-InstructV0.2 Modell
    V0.2.7.x - Bessere, aber langsamere Embeddings, sortieren des nähesten Context nach Datum
    V0.2.6.x - Einbeziehen von Datum in Quellen
    V0.2.5.x - Opimieren question-Vorverarbeitung, mehr Parameter, ignorieren großer chromaDB Distanzen, Schätzung Initialantwortzeit
    V0.2.x - komplette Neuadaption von V0.1, völlig OHNE langchain und mit optimiertem Abfragetext
    V0.1.x - pure langchain basierte Adaption von privateGPT, detuscher Prompt
"""

"""
Initial banner Message
"""
print("\nPCnewsGPT Wissensabfrage V0.2.9\n")

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
max_resort_distance = float(os_environ.get('MAX_RESORT_DISTANCE',0.1))
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

# Central prompt template (note: substituted context has \n at end) and system-prompt prefix/suffix
# needs updates for different models
prompt_template = 'Es folgt eine Liste von Informationen als Kontext, sortiert nach aufsteigender Wichtigkeit:\n{}' + \
                'Anweisung: Beantworte die folgende Frage kurz, genau, und in diesem Kontext.\n' + \
                'Frage: {}?\nAntwort: '
# Mistral-style instruct: <s>[INST] {prompt} [/INST]
system_prompt_prefix = ''  # '<s>[INST] ' # only for EN dialogues
system_prompt_suffix = ''  # ' [/INST]'   # only for EN dialogues

"""
main query loop - interactive questions and answers until empty line is entered
"""
from operator import itemgetter as operator_itemgetter
while True:
    # *** get user question (empty line exits) and clean it up (also remove ? or . at end, it confuses some LLMs)
    question = input("\n### Neue Frage: ").strip().rstrip('?').rstrip('.').strip() 
    if question == "":
        break

    # *** create embeddings and ask DB for context
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
        date = metadata.get('creationDate','D:19000101').lower()    # default to 0000-01-01 if no date is given
        context_dates.append(f'{date[2:6]}-{date[6:8]}-{date[8:10]}' if date.startswith('d:') else date)
    
    # *** generate LLM prompt context
    # this assumes:
    #   that the context_text items came already cleaned up from the DB
    #   that the DB returns items in order of relevance
    # newer elements, below max_resort_distance from nearest element, are sorted to the beginning
    context = []
    if len(context_distances)>0:
        resort_dist=min(context_distances)+max_resort_distance
        for i in range(len(context_texts)):
            # it ignores context information which is too far away
            if context_distances[i] <= max_context_distance:
                # insert at end, if 1st element or above important_dist
                if (len(context)==0) or (context_distances[i] > resort_dist):
                    context.append(i)
                # else check date, insert at beginning if newer, else append
                else:
                    date_i = int(context_dates[i][0:4])*10000 + int(context_dates[i][5:7])*100 + int(context_dates[i][8:10])
                    date_1 = int(context_dates[context[0]][0:4])*10000 + int(context_dates[context[0]][5:7])*100 + int(context_dates[context[0]][8:10])
                    if date_i > date_1:
                        context.insert(0,i)
                    else:
                        context.append(i)

    # generate context string (newest at end)
    context_text=""
    for i in range(len(context)-1,-1,-1):
        # numbered: context_text += f"{i+1}. '{context_texts[context[i]]}'\n"
        context_text += f"'{context_texts[context[i]]}',\n"

    # *** generate LLM prompt only if we have viable prompt context
    if context_text == "":
        print(f"\n### Keine passenden Informationen in Wissensbasis gefunden.")
    else:
        prompt = system_prompt_prefix + prompt_template.format(context_text, question) + system_prompt_suffix

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
        print('')
        # Model automatically prints statistics after answer if not hide_source
    
    # Print sources
    if not hide_source:
        print(f"\n### Quellen:")
        for i in range(len(context)):
            metadata = context_metadatas[context[i]]
            print(f"[{i+1}] {metadata.get('source','').split('/')[-1]} Seite:{metadata.get('page','-')} Textteil:{metadata.get('chunk','-')} Distanz:{context_distances[context[i]]:.2f} Datum:{context_dates[context[i]]}", end="")
            if not hide_source_details:
            # dont print real \n in source texts
                print(" :\n'"+context_texts[context[i]].replace("\n", "\\n")+"'")
            else:
                print("")
            continue
        for i in range(len(context),len(context_texts)):
            metadata = context_metadatas[i]
            print(f"[{i+1}] Ignoriert:{metadata.get('source').split('/')[-1]} Seite:{metadata.get('page','-')} Textteil:{metadata.get('chunk','-')} Distanz:{context_distances[i]:.2f}")
