"""
*** PCnewsGPT Wissensimporter - import.py ***

    + PCnews PDFs get converted into text via pdfminer
    + clean PDFs
      + get rid of strange/redundant \n characters
      + substitute ligatures with processable text
      + get rid of other strange things
    + split PDFs into pages for better chunking
    + split pages into digestable chunks of text (SpaCy, text-cleanup)
    + embed and persist in chromadb
    + optimized for quality first. Speed might come later.
    
    Update V0.1.2: besser lesbare, verständliche chunk-texte, Tippfehler korrigiert
        
"""

"""
Load Parameters, etc.
"""
from dotenv import load_dotenv
from os import environ as os_environ
from ast import literal_eval
load_dotenv()
persist_directory = os_environ.get('PERSIST_DIRECTORY','db')
embeddings_model_name = os_environ.get('EMBEDDINGS_MODEL_NAME','paraphrase-multilingual-mpnet-base-v2')
source_directory = os_environ.get('SOURCE_DIRECTORY', 'source_documents')
text_splitter_name = os_environ.get('TEXT_SPLITTER','RecursiveCharacterTextSplitter')
text_splitter_parameters = literal_eval(os_environ.get('TEXT_SPLITTER_PARAMETERS','{"chunk_size": 500, "chunk_overlap": 50}'))

"""
Initial banner Message
"""
print("\nPCnewsGPT Wissensimporter V0.1.1\n")

"""
Map file extensions to document loaders and their arguments
"""
from langchain.docstore.document import Document as langchain_Document
from langchain.document_loaders import (
    PDFMinerLoader,
    TextLoader
)
loader_mappings = {
    ".pdf": (PDFMinerLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"})
}

"""
Initialize Text Splitter
"""
# dynamically import the langchain text splitter class and instantiate it
from importlib import import_module
text_splitter_module = import_module("langchain.text_splitter")
TextSplitter = getattr(text_splitter_module, text_splitter_name)
text_splitter = TextSplitter(**text_splitter_parameters)

"""
Initialize Embeddings
"""
from langchain.embeddings import HuggingFaceEmbeddings
print(f"Embeddings {embeddings_model_name} werden eingelesen...\n")
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

"""
Initialize ChromaDB
"""
from langchain.vectorstores import Chroma
from chromadb.config import Settings as Chroma_Settings
# Define the Chroma settings
chroma_settings = Chroma_Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
)

"""
parse source_directory for all filenames to load
"""
from os import path as os_path
from glob import glob
file_paths = []
for ext in loader_mappings:
    file_paths.extend(
        glob(os_path.join(source_directory, f"**/*{ext}"), recursive=True)
    )

"""
Load and convert file_path into a langchain document
"""
from regex import sub as regex_sub
def load_file(file_path: str) -> langchain_Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    assert ext in loader_mappings       # file_paths should alread only be supported types
    # load via langchain
    loader_class, loader_args = loader_mappings[ext]
    loader = loader_class(file_path, **loader_args)
    lc_doc = loader.load()
    assert len(lc_doc) == 1             # should only be one document per file up til now

    # Tidy up PDFs
    if ext == ".pdf":
        doc = lc_doc[0].page_content   #file_path still holds document source
        # change single \n in content to " ", but not multiple \n
        doc = regex_sub(r'(?<!\n)\n(?!\n)', ' ',doc)
        # remove line-break remnants
        doc =doc.replace('- ', '')
        # change multiple consecutive \n in content to just one \n
        doc = regex_sub(r'\n{2,}', '\n',doc)
        # substitute known ligatures
        doc =doc.replace('(cid:297)', 'fb')
        doc =doc.replace('(cid:322)', 'fj')
        doc =doc.replace('(cid:325)', 'fk')
        doc =doc.replace('(cid:332)', 'ft')
        doc =doc.replace('(cid:414)', 'tf')
        doc =doc.replace('(cid:415)', 'ti')
        doc =doc.replace('(cid:425)', 'tt')
        doc =doc.replace('(cid:426)', 'ttf')
        doc =doc.replace('(cid:427)', 'tti')
        # substiture strange characters
        doc =doc.replace('€', 'EUR')
        doc =doc.replace("„", '"')
        doc =doc.replace('\uf0b7', '*')
        doc =doc.replace('\uf031\uf02e', '1.')
        doc =doc.replace('\uf032\uf02e', '2.')
        doc =doc.replace('\uf033\uf02e', '3.')
        doc =doc.replace('\uf034\uf02e', '4.')
        doc =doc.replace('\uf035\uf02e', '5.')
        doc =doc.replace('\uf036\uf02e', '6.')
        doc =doc.replace('\uf037\uf02e', '7.')
        doc =doc.replace('\uf038\uf02e', '8.')
        doc =doc.replace('\uf039\uf02e', '9.')
                
        # split doc-content into pages andremove trailing empty pages
        pages =doc.split('\x0c')
        while pages[-1] == '':
            pages.pop()
        # if there are no remaining pages, empty the text in lc_doc
        if len(pages) == 0:
            lc_doc[0].page_content = ""
        # if its just one page, update lc_doc with processed content
        elif len(pages) == 1:
            lc_doc[0].page_content = pages[0]
        # if there are muliple pages, create new lc_doc from the pages
        else:
            lc_doc.pop()
            pg_num=1
            for page in pages:
                lc_doc.append( langchain_Document(
                    page_content = page,
                    metadata = {"source": f"{file_path} (page {pg_num})"}
                ))
                pg_num += 1

    # return the loaded doc               
    return lc_doc


"""
Load + process all documents in source_directory
"""
db = None
total_chunks=0
# load all the files
print(f"Dokumentdateien in {source_directory} werden eingelesen und verarbeitet...\n")
for idx,file_path in enumerate(file_paths):
    print(f"Datei {file_path} ({idx+1}/{len(file_paths)})...")
    documents=load_file(file_path)                  # txt as 1 document, pdfs as 1 document per page
    print(f"... wurde eingelesen und in {len(documents)} Seite(n) umgewandelt ...")
    # Split into chunks of text
    chunks = text_splitter.split_documents(documents)
    print(f"... zerteilt auf {len(chunks)} Textteil(e) ...")
    total_chunks += len(chunks)
    # tidying-up chunk text and numbering of chunks per source
    n_chunk=1
    chunk_name=chunks[0].metadata["source"]
    for i in range(len(chunks)):
        # new chunk name if source is different
        if chunk_name!=chunks[i].metadata["source"]:
            n_chunk=1
            chunk_name=chunks[i].metadata["source"]
        else:
            n_chunk += 1
        chunks[i].metadata["source"]= f'{chunks[i].metadata["source"]} (chunk {n_chunk})'
        
        # Tidying-up chunk text
        txt = chunks[i].page_content.replace('\n', ' ') # remove all \n from text
        txt = txt.replace("'", '"')                     # replace single with double quotes
        txt = ' '.join(txt.split())                     # replace multiple spaces with single space
        txt = chunks[i].page_content
        
    # create embeddings and persist
    if db is None:
        # creeate new db
        db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory, client_settings=chroma_settings)
    else:
        # add to existing db
        db.add_documents(chunks)
    db.persist()
    print("... und in der Wissensdatenbank gespeichert.\n")
    
# Statistics
print(f"Insgesamt {len(file_paths)} Dokument(e) mit {total_chunks} Textteil(en) wurden aus dem Ordner {source_directory} eingelesen.")
db = None
